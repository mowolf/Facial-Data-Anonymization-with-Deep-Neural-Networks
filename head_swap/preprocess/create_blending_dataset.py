import json
import random
from pathlib import Path

import torch
from PIL import Image
import click
import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from postprocess.get_metrics import MetricVisualizer, get_identity_loss
from preprocess.enhance_masks import dilate
from preprocess.face_detector import calculate_tight_bounding_box, save_cutout_with_margin
from preprocess.utils import get_files, make_color_transparent, get_area_with_color, get_subfolders
from preprocess.transform import apply_transforms, random_rotate


def _files_to_dataset(files: Path, save_path: str, mask_segmentation_path: str,
                      border_perc: float = 0.1, margin: float = 0.33, out_resolution: int = 256,
                      with_segmentation: bool = False, real_face: bool = True,
                      do_fill_with_mean_color: bool = False, do_transform: bool = True, segmentation_path: str = ".",
                      frame_distance: int = 1, fake_face_folder: str = ''):
    """
    This function takes a file list as input and saves the dataset to save_path. Returns error count

    :param frame_distance: 1/frame_distance of images are processed with a distance of frame_distance - 1
    :param segmentation_path: path to segmentation files
    :param do_transform: transform heads
    :param do_fill_with_mean_color: fill replaced background with mean color instead of black
    :param with_segmentation: use segmentation not bounding box
    :param out_resolution: output resolution
    :param files: files to process
    :param save_path: folder to save output
    :param border_perc: Add a border to the replaced region, border = border_perc * max(width,height)
    :param margin: Add margin to bbox, margin_added = margin * edge_length
    :return: err_count, how many files needed to be discarded when processing due to size limits
    """
    err_cnt = 0
    i = 0

    ########
    # If you want to process images where you already have extracted the face keypoints you can use this
    from_data = False
    if from_data:
        print("\nReading .json")
        global loaded_keypoint_data
        with open('/home/mo/experiments/masterthesis/flickr/ffhq-dataset-v2.json', 'r') as f:
            loaded_keypoint_data = json.load(f)
        print("Processing data")
    ########
    # if program breaks
    # saved_files = [x.stem for x in get_files("/mnt/raid5/mo/ffhq_wide", file_type=".png")]

    for file in tqdm(files):
        # only keep every _frame_distance_ image
        if i % frame_distance != 0 and i > 1:
            continue
        i += 1
        # if file.stem in saved_files:
        #     continue
        # try:
        original_img = cv2.imread(str(file), 1)
        img_to_transform = original_img.copy()

        err_cnt = _save_blending_pair(original_img, img_to_transform, file, save_path, err_cnt, mask_segmentation_path,
                                      border_perc=border_perc,
                                      margin=margin,
                                      out_resolution=out_resolution,
                                      with_segmentation=with_segmentation,
                                      real_face=real_face,
                                      do_transform=do_transform,
                                      do_fill_with_mean_color=do_fill_with_mean_color,
                                      segmentation_path=segmentation_path,
                                      fake_face_folder=fake_face_folder
                                      )
        # except AttributeError:
        #     print("Corrupted Image")
        #     err_cnt += 1
    return err_cnt


def _is_mask_valid(mask):
    """
    Checks if mask is valid
    :param mask: mask file
    :return: bool
    """
    # check if face mask is big enough
    if np.sum(mask > 0) < (mask.shape[0] * mask.shape[1]) / 5:
        return False
    # check if there is face and hair in the mask
    elif not ((mask == 1).any() and (mask == 12).any()):
        return False
    else:
        return True


def _save_blending_pair(original_img, img_to_transform, file, save_path, err_cnt, mask_segmentation_path,
                        border_perc=0.1,
                        margin=0.33,
                        out_resolution=256, with_segmentation=False, real_face=True, do_transform=True,
                        do_fill_with_mean_color=True, segmentation_path: str = ".", fake_face_folder: str = ""):
    """
    Main function that processes one file given the specifications
    """
    ####
    # Change the filename if it is differently named in the location
    # name = Path(file).parent.stem + "_" + Path(file).stem
    ####
    name = Path(file).stem

    if not with_segmentation:
        face_keypoints = loaded_keypoint_data[file.stem]['in_the_wild']['face_landmarks']
        err_cnt = _process_file_with_bbox(border_perc, do_transform, err_cnt, img_to_transform, margin, name,
                                          original_img, out_resolution, save_path, face_keypoints)
        return err_cnt
    else:
        # get segmentation mask from files
        try:
            mask = get_segmentation_mask_from_name(file.name, segmentation_path)
        except FileNotFoundError:
            print(f"Mask missing for {file.name}")
            return 1
        # check mask to contain face & background region
        if not _is_mask_valid(mask):
            return False
        try:
            _process_file_with_segmentation(do_fill_with_mean_color, do_transform, file, mask, name, original_img,
                                            real_face, save_path, mask_segmentation_path, fake_face_folder)
        except cv2.error:
            print("Cv2 error in _process_file_with_segmentation")
            return 1
        return 0


def _process_file_with_segmentation(do_fill_with_mean_color, do_transform, file, mask, name, original_img, real_face,
                                    save_path, mask_segmentation_path, fake_face_folder):
    """
    Processes a single file
    :param do_fill_with_mean_color: Fill background with mean color instead of black
    :param do_transform: randomly rotate in range of -10° to 10°
    :param file: file to process
    :param mask: corresponding masl
    :param name: image name
    :param original_img: image corresponding to file
    :param real_face: bool, if we want to swap with
    :param save_path:
    :param mask_segmentation_path:
    """

    # Use first image
    if original_img.shape[0] != original_img.shape[1]:
        original_img = original_img[:, :256, :]
    # get segmentation results
    face, background, foreground_mask, background_mask = get_face_and_background(original_img,
                                                                                 mask,
                                                                                 mask_segmentation_path,
                                                                                 Path(fake_face_folder),
                                                                                 real_face=real_face,
                                                                                 name=file)
    if face is not None:
        if do_transform:
            face = transform_face(face, background_mask)
        else:
            # convert to pil
            face = cv2_to_pil(face)
        if do_fill_with_mean_color:
            # fill background with mean color of region
            background = fill_with_mean_color(original_img, background, foreground_mask, background_mask)
        # paste face onto background
        result = insert_face(background, face)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        merged = np.concatenate((np.asarray(original_img), result), 1)
        merged = Image.fromarray(merged)
        merged.save(f"{save_path}/{name}.png")


def _process_file_with_bbox(border_perc, do_transform, err_cnt, img_to_transform, margin, name, original_img,
                            out_resolution, save_path, face_keypoints):
    """
    If not using segmentation, we can just use the bbox
    :param border_perc:
    :param do_transform:
    :param err_cnt:
    :param img_to_transform:
    :param margin:
    :param name:
    :param original_img:
    :param out_resolution:
    :param save_path:
    :param face_keypoints:
    :return:
    """
    i = 0

    ########
    # gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # dets = detector(gray, 1)
    ########
    # for det in dets:
    for _ in range(1):
        ########
        # keypoints = get_face_keypoints(original_img, det)
        # (x, y, w, h) = calculate_tight_bounding_box(keypoints)
        ########
        (x, y, w, h) = calculate_tight_bounding_box(np.asarray(face_keypoints))
        # remove small faces
        if w < 50 or h < 50:
            continue

        # original_replaced = original_img.copy()
        try:
            # get img_to_transform and apply transform
            # y_face_margin = int(margin * h)
            # x_face_margin = int(margin * w)
            ####
            # make square
            transformed_img = save_cutout_with_margin(img_to_transform, margin, x, y, w, h)

            final = cv2.resize(transformed_img, (out_resolution, out_resolution), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{save_path}/{name}.png", final)

            #######
            # if do_transform:
            #     transformed_img = apply_transforms(transformed_img)
            # Replace Face in future image with previous frame
            # original_replaced = _replace_face(x, y, w, h, original_replaced, transformed_img, x_face_margin,
            #                                   y_face_margin)
            # Add a border to the replaced region
            # h_new, w_new, _ = transformed_img.shape
            # edge = max(h_new, w_new)
            # original_replaced = _add_border(original_replaced, x, y, w, h, border_perc, edge, x_face_margin, y_face_margin)
            # final = _add_border(original_img, x, y, w, h, border_perc, edge, x_face_margin, y_face_margin)
            # flip randomly
            # if random.getrandbits(1):
            #     original_cut = flip_image(original_cut)
            #     original_replaced = flip_image(original_replaced)
            # merged = np.concatenate((final, original_replaced), axis=1)
            # merged = cv2.resize(merged, (2 * out_resolution, out_resolution), interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite(f"{save_path}/{name}_{i}.png", merged)
            #######

            i += 1
        except (ValueError, cv2.error, IndexError, TypeError, AttributeError):
            # This rather bad statemt catches multiple errors of this implementation which does not take care of
            # some edge cases but errors at less than 5% of data which is enough for my use case
            err_cnt += 1
            continue
    return err_cnt


def _add_border(img, x, y, w, h, border_perc, edge, x_face_margin, y_face_margin):
    """
    gets bbox with border and margin around x,y,w,h
    :param img: image
    :param x: x
    :param y: y
    :param w: w
    :param h: h
    :param border_perc: border = border_perc * edge
    :param edge: length of edge
    :param x_face_margin: x margin
    :param y_face_margin: y margin
    :return: curout image
    """
    if border_perc > 0:
        border = int(border_perc * edge)
        img = img[y - y_face_margin - border:y + h + y_face_margin + border,
              x - x_face_margin - border:x + w + x_face_margin + border,
              :]
    return img


def _replace_face(x, y, w, h, original, replace, x_face_margin, y_face_margin, jitter_factor=0.7):
    """
    Replaces x,y,w,h in original with replace image
    :param x: x bbox
    :param y: y
    :param w: w
    :param h: h
    :param original_replaced: image to replace head
    :param transformed_img: image to place onto original
    :param x_face_margin: margin x
    :param y_face_margin: margin y
    :param jitter_factor: factor by what insertion is noisy
    :return: Image with inserted face
    """
    x_jitter = random.randint(- int(x_face_margin * jitter_factor), int(x_face_margin * jitter_factor))
    y_jitter = random.randint(- int(y_face_margin * jitter_factor), int(y_face_margin * jitter_factor))
    original[y - y_face_margin + y_jitter:y + h + y_face_margin + y_jitter,
    x - x_face_margin + x_jitter:x + w + x_face_margin + x_jitter,
    :] = replace

    return original


def fill_with_mean_color(original_img, background, foreground_mask, background_mask):
    """
    Fills background with mean of original_img(foreground_mask) in region of foreground_mask
    and original_img(background_mask) in region of background_mask
    :param original_img: cv2 img
    :param background: cv2 img, cutout face
    :param foreground_mask: boolean mask
    :param background_mask: boolean mask
    :return: background image with recolored black areas
    """
    black_region = get_area_with_color(background, (0, 0, 0))
    background_mask = background_mask & black_region

    # make 1d masks 3d
    foreground_mask_3d = foreground_mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    background_mask_3d = background_mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]

    # get mean color
    foreground_color = np.mean(np.ma.masked_array(original_img, 1 - foreground_mask_3d), axis=(0, 1))
    background_color = np.mean(np.ma.masked_array(original_img, 1 - background_mask_3d), axis=(0, 1))

    # fill foreground
    background[..., :][foreground_mask] = np.asarray(foreground_color, dtype=np.uint8)
    # fill background
    background[..., :][background_mask] = np.asarray(background_color, dtype=np.uint8)

    return background


def insert_face(img, face, add_border=True):
    """
    inserts face onto image
    :param add_border: add black border around face
    :param img: Image to paste onto
    :param face: face image with removed background
    :return: merged images
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = make_color_transparent(face, (0, 0, 0))
    if add_border:
        black = np.array(face)
        black[..., :3] = [0, 0, 0]
        dilated = dilate(black, 7)
        dilated = Image.fromarray(dilated)
        dilated.paste(face, (0, 0), face)
        face = dilated

    img = Image.fromarray(img)
    img.paste(face, (0, 0), face)
    return img


def transform_face(face, background_mask):
    """
    Transforms face by rotating and removing the background
    :param face: fill image
    :param background_mask: mask
    :return: image with just the face
    """
    # remove background
    face[..., :][background_mask] = (0, 0, 0)
    # transform
    face = apply_transforms(face, no_border=True, no_color=True, no_contrast=True).copy()
    # resize
    face = cv2.resize(face, (256, 256))
    # convert to cv2
    face = cv2_to_pil(face)
    if 1 > random.getrandbits(1):
        face = random_rotate(face)
    return face


def cv2_to_pil(image):
    """
    Converts cv2 img to pil
    :param image: cv2 image
    :return: pil image
    """
    image = image.astype(dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


def get_head_size_from_keypoints(keypoints):
    """
    Returns head size and head angle
    :param keypoints: facial keypoints (dlib)
    :return: distance of left_to_right, nose_to_head and the tilted head angle
    """
    keypoints = np.asarray(keypoints)
    try:
        left_to_right = abs(np.linalg.norm(keypoints[16] - keypoints[0]))
        nose_to_head = abs(np.linalg.norm(keypoints[27] - keypoints[8])) * 1.8

        # compute the angle
        dY = keypoints[27][1] - keypoints[8][1]
        dX = keypoints[27][0] - keypoints[8][0]
        angle = abs(np.degrees(np.arctan2(dY, dX)))

    except IndexError:
        left_to_right, nose_to_chin = 0, 0
        angle = 0

    return left_to_right, nose_to_head, angle


def get_matching_head_file(file: Path):
    """
    Returns the name of a matching file.
    Matching is done via the metrics:
    - keypoints
    - age
    - gender
    - identity
    :param file: Path to the file
    :return: str, name of the matching file
    """
    # get metric of file
    if file.name in metrics_real:
        truth = metrics_real[file.name]
    else:
        truth_metrics = m.get_metric_for_file(file)
        truth = truth_metrics[file.name]
    # calculate width and angle
    w, h, angle = get_head_size_from_keypoints(truth["dlib_keypoints"])
    if min(w, h, abs(angle)) == 0:
        return None, None
    # get age and gender
    age = float(truth["age"])
    interval = 5 if age < 18 else 10
    male2female = float(truth["male2female"])
    truth_keypoints = np.asarray(truth["dlib_keypoints"]) - np.asarray(truth["dlib_bbox"][0][:2])

    # random shuffle how we iterate through dict
    keys = list(metrics_fake.keys())
    random.shuffle(keys)

    for key in keys:
        # do not take same file for head swap
        if key == file.name or key[:3] == "000" or key[0:3] == "011":
            continue
        # calculate width and angle
        w_, h_, angle_ = get_head_size_from_keypoints(np.asarray(metrics_fake[key]["dlib_keypoints"]))
        # check that not equal zero
        if min(w_, h_, abs(angle_)) == 0:
            continue

        keypoints_to_test = np.asarray(truth["dlib_keypoints"]) - np.asarray(truth["dlib_bbox"][0][:2])
        max_keypoint_dist = np.linalg.norm(np.amax(keypoints_to_test - truth_keypoints, axis=0))
        # calculate ratios
        with_ratio = min(w, w_) / max(w, w_)
        scale = h / h_
        # check that all if conditions hold to find a match
        if with_ratio > 0.90:
            if (min(angle, angle_) / max(angle, angle_)) > 0.90:
                if max_keypoint_dist < 20:
                    male2female_ = float(metrics_fake[key]["male2female"])
                    if (male2female > 0.5 and male2female_ > 0.5) or (male2female < 0.5 and male2female_ < 0.5):
                        age_ = float(metrics_fake[key]["age"])
                        if abs(age - age_) <= interval:
                            identity_loss = get_identity_loss(torch.Tensor(metrics_fake[key]["identity"]),
                                                              torch.Tensor(truth["identity"]))
                            if identity_loss > 0.5:
                                # print(f"{file.name} - {angle} vs {angle_}\n")
                                # print(f"angle: {angle} vs {angle_}\n")
                                # print(f"width: {w} vs {w_}\n")
                                # print(f"height: {h} vs {h_}\n")
                                # print(f"age: {age} vs {float(metrics[key]['age'])}\n")
                                return key, scale
    print(f"{file.name} not matched")
    return None, None


def get_matching_face_and_segmentation(path: Path, name: str,
                                       segmentation_path: str,
                                       random: bool = False):
    """
    Returns the segmentation mask and the corresponding image to euther a random or a matching image
    :param path: Path to Parent Directory of the file location
    :param name: name of the file you want to find a match
    :param segmentation_path: path where the segmentation masks are saved
    :param random: if True returns a random sample
    :return:
    """

    # test set range from 000* to 010*
    max_tries = 2
    i = 0
    valid = False
    while not valid:
        if random:
            files = get_files(path, ".png")
            file = random.choice(files)
            scale = 1
        else:
            # use metrics to find best head file
            file, scale = get_matching_head_file(name)
            if file is None:
                return None, None, None
            file = path / file
        mask = get_segmentation_mask_from_name(file.name, segmentation_path)
        i += 1

        if i >= max_tries:
            valid = True
        else:
            valid = _is_mask_valid(mask)

    file = path / file.name
    face = cv2.imread(str(file), 1)
    # enable if you read from pix2pix datasets
    assert face is not None, f"{path / file.name} not found."

    if face.shape[0] != face.shape[1]:
        face = face[:, :256, :]

    return face, mask, scale


def get_face_and_background(original_img, mask, mask_segmentation_path, face_path, real_face=True, name=""):
    """
    Returns Face and background segmentation
    :param original_img: original image
    :param mask: full face segmentation mask based on celebA HQ labels
    :param segmentation_path: path to seg masks as image
    :param face_path: path to image
    :param real_face: bool if we want to use the real face or a matching face
    :param name: name of the file
    :return: face, baclground as images and foreground_mask, background_mask as bool
    """
    background_mask, foreground_mask = split_background_foreground_segmentation(mask, erode_size=40)

    if real_face:
        face = original_img.copy()
        scale = 1
    else:
        # get face and mask from file range
        face, second_mask, scale = get_matching_face_and_segmentation(face_path, name,
                                                                      segmentation_path=mask_segmentation_path)
        if face is None:
            return None, None, None, None
        # NOTE: Depending on the erode_size masks need to be adjusted:
        # erode_size=40 -> 0.3, 0,2;
        # erode_size=20 ->  0.2, 0.1
        background_mask, _ = split_background_foreground_segmentation(second_mask, erode_size=40)

    face[..., :][background_mask] = (0, 0, 0)

    # scle if needed
    diff = int((scale * 256 - 256) / 2)
    if abs(diff) > 20:
        face = cv2.resize(face, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if diff < 0:
            border = - diff
            face = cv2.copyMakeBorder(face, border, border, border, border, cv2.BORDER_CONSTANT, None, [0, 0, 0])
        else:
            face = face[diff:-diff, diff:-diff, :]

    try:
        y_min, y_max = min(np.where(foreground_mask == True)[0]), max(np.where(foreground_mask == True)[0])
        x_min, x_max = min(np.where(foreground_mask == True)[1]), max(np.where(foreground_mask == True)[1])
    except ValueError:
        y_min = 0
        y_max = 0
        x_min = 0
        x_max = 0
        print(f"{name} missed valid_mask check?!")

    y_min_padding = int(y_min * 0.3)
    y_max_padding = int((mask.shape[0] - y_max) * 0.3)
    x_min_padding = int(x_min * 0.2)
    x_max_padding = int((mask.shape[1] - x_max) * 0.2)

    background = original_img.copy()
    background[y_min - y_min_padding:y_max + y_max_padding, x_min - x_min_padding:x_max + x_max_padding, :] = (0, 0, 0)

    return face, background, foreground_mask, background_mask


def split_background_foreground_segmentation(mask, erode_size=20, dilate_size=20):
    """
    Returns a foreground background segmentation from the multi segmentation of celebaHQ
    :param mask: input mask
    :param erode_size: size to erode the boundary
    :param dilate_size: size to dilate the boundray
    :return: background and foreground mask (which are the exact opposites)
    """
    foreground_mask = mask > 0
    remove_mask = mask < 13
    foreground_mask = foreground_mask & remove_mask
    kernel = np.asarray([[True] * erode_size] * 3)
    foreground_mask = ndimage.binary_erosion(foreground_mask, structure=kernel).astype(foreground_mask.dtype)
    kernel = np.asarray([[True] * dilate_size] * 3)
    foreground_mask = ndimage.binary_dilation(foreground_mask, structure=kernel).astype(foreground_mask.dtype)
    background_mask = np.logical_not(foreground_mask)
    return background_mask, foreground_mask


def get_segmentation_mask_from_name(name: str, seg_path: str, verbose: bool = False):
    """
    Returns the mask image given the name and the segmentation_path
    :param name: name of the file
    :param seg_path: path to seg makss
    :param verbose: print what is opened
    :return: mask image
    """
    if verbose:
        print(f"Opening {name} in {seg_path} to retrieve segmentation mask.")
    seg_path = Path(seg_path)
    mask = Image.open(seg_path / name)
    mask = np.asarray(mask)

    return mask


# Face Forensics
# AV SPEECH WITH LARGER BBoxes /mnt/avspeech_extracted/dlib_extracted/video_crops/
# AV SPEACH SETI /mnt/raid5/sebastian/avspeech_extracted_seti/avspeech_extracted/dlib_extracted/video_crops
# AV SPEECH MORIA /mnt/raid5/sebastian/avspeech_extracted_moria/avspeech_extracted/dlib_extracted/video_crops
@click.command()
@click.option('--data_path',
              help='Data root path.')
@click.option('--save_path',
              help='Path where to save images')
@click.option('--file_type', default='.png',
              help='File type of images to process.')
@click.option('--frame_distance', default=1,
              help='Distance between frames to sample pairs.')
@click.option('--with_segmentation', is_flag=True,
              help='Use segmentation masks instead of face detection.')
@click.option('--use_fake_face', is_flag=True,
              help='Paste head of random test set person/file onto file - only for test set generation!.')
@click.option('--fake_face_folder', default="",
              help='Location of folder with faces to use.')
@click.option('--do_not_transform', is_flag=True,
              help='Transform images.')
@click.option('--do_fill_with_mean_color', is_flag=True,
              help='Fill replaced areas with mean colors.')
@click.option('--margin', default=0.33,
              help='Add margin to extracted face bounding box.')
@click.option('--border_perc', default=0.1,
              help='Add a border to the replaced region')
@click.option('--out_resolution', default=256,
              help='Height of output image')
@click.option('--segmentation_path',
              default="",
              help='Folder of masks belonging to data_path files')
@click.option('--mask_segmentation_path',
              default="",
              help='Folder of masks belonging to fake_face_folder')
def create_blending_dataset(data_path: str,
                            file_type: str,
                            frame_distance: int,
                            out_resolution: int,
                            with_segmentation: bool,
                            use_fake_face: bool,
                            fake_face_folder: str,
                            border_perc: float,
                            margin: float,
                            save_path: str,
                            do_not_transform: bool,
                            do_fill_with_mean_color: bool,
                            segmentation_path: str,
                            mask_segmentation_path: str,
                            search_subfolders: bool = False
                            ):
    """
    Create a blending dataset, where we remove the bounding box of the head and insert the old head again
    (but transformed) to learn the merging and blending. For test set generation you can also swap the head.
    """
    if use_fake_face:
        print("No transformations.")
        do_not_transform = True
    else:
        print("Replacing with same data.")
        mask_segmentation_path = segmentation_path
        fake_face_folder = data_path

    if use_fake_face:
        print("-- Loading Metrics--\n")
        global metrics_fake, metrics_real, m
        m = MetricVisualizer()
        assert fake_face_folder != "", "No folder given for fake faces"
        metrics_fake = m.calc_metric_for_folder(folder=Path(fake_face_folder))
        metrics_real = m.calc_metric_for_folder(folder=Path(data_path))


    err_cnt = 0
    # load all files
    print("-- Loading Files--\n")
    if search_subfolders:
        print("searching for subfolders")
        folders = get_subfolders(data_path, verbose=True)
    else:
        folders = [Path(data_path)]

    print("-- Processing Files--\n")
    for folder in tqdm(folders):
        files = get_files(Path(data_path) / folder, file_type)
        files.sort()
        # get corresponding pairs
        err_cont_ = _files_to_dataset(files, save_path, mask_segmentation_path,
                                      border_perc=border_perc,
                                      margin=margin,
                                      out_resolution=out_resolution,
                                      with_segmentation=with_segmentation,
                                      real_face=not use_fake_face,
                                      do_fill_with_mean_color=do_fill_with_mean_color,
                                      do_transform=not do_not_transform,
                                      segmentation_path=segmentation_path,
                                      fake_face_folder=fake_face_folder,
                                      frame_distance=frame_distance)
        err_cnt += err_cont_

    print(f"Total errors: {err_cnt}")


if __name__ == '__main__':
    create_blending_dataset()
