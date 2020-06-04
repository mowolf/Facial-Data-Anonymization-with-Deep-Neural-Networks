import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
import click

from preprocess.utils import recolor_to_rgba, make_color_transparent, merge_images_side_by_side, get_segments, get_files, create_folder_structure

def im_resize(img):
    """
    Resizes image to 256x256px
    :param img: loaded PIL image
    _:return resized pil image
    """
    return img.resize((256, 256), Image.NEAREST)


def add_warp(img, dilate=0, erode=0, deg=0, c=0, f=0):
    """
    Adds a warp to the image
    """
    img = img.filter(ImageFilter.MaxFilter(2 * dilate + 1))
    img = img.filter(ImageFilter.MinFilter(2 * erode + 1))

    if deg < 0:
        deg = 360 + deg
    img = img.rotate(deg)

    a = e = 1
    b = d = 0
    img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))

    return img


def add_random_warp(img, seed=None):
    """
    Adds a random warp to the image
    """
    if not seed:
        dilate = 0
        erode = 0
        if random.getrandbits(1):
            if random.getrandbits(1):
                dilate = random.randint(1, 3)
                erode = 0
            else:
                dilate = 0
                erode = 1

        deg = random.randint(0, 5)
        if random.getrandbits(1):
            deg = 360 - deg
    else:
        dilate = seed[0]
        if dilate:
            dilate = random.randint(1, 5)
        erode = seed[1]
        if erode:
            erode = 1
        move = seed[2]
        deg = seed[3]

    c = 0
    f = 0
    img = add_warp(img, dilate=dilate, erode=erode, deg=deg, c=c, f=f)

    return img


def render_frames(frames: list):
    """
    Renders list of PIL or openCV images
    :type frames: object
    :return merged image of same type as imput image (PIL or openCV)
    """
    opencv = False
    try:
        # assume Opencv image
        shape = (frames[0].shape[0], frames[0].shape[1])
        opencv = True
    except AttributeError:
        # we dont have a opencv image
        pass
    if not opencv:
        # so we have a PIL image
        shape = frames[0].size

    background = Image.new("RGBA", shape, "BLACK")

    for frame in frames:
        if opencv:
            frame = Image.fromarray(frame)

        frame = make_color_transparent(frame, (0, 0, 0))
        background.paste(frame, (0, 0), frame)

    if opencv:
        return np.array(background)
    else:
        return background


def create_mask_from_id(pic_id: int,
                        face_only_segmentation=False,
                        warp=False,
                        showGUI=False,
                        data_path=Path('/home/mo/datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno'),
                        skin=True, nose=True, glasses=True, eyes=True, brows=True, ears=True,
                        mouth=True, u_lip=True, l_lip=True, hair=True, neck=True, misc=True,
                        ):
    """Creates one image with all segmentation masks
    :param showGUI: displays hui
    :type data_path: Path
    :param warp: warps the masks randomly
    :param face_only_segmentation: only face
    :param pic_id: id of image
    :param data_path: path object of data root containing masks
    :return: the merged Image object
    """
    sub_folder = str(pic_id // 2000)
    pic_id = str(pic_id).zfill(5)

    segments = get_segments()
    unprocessed_segments = 0
    frames = []
    warped_frames = []

    # seed for warping
    if warp:
        deg = random.randint(0, 10)
        if random.getrandbits(1):
            deg = 360 - deg
        seed = [random.getrandbits(1), random.getrandbits(1), random.getrandbits(1), deg]

    # render loop
    for key, segment in segments.items():

        if face_only_segmentation:
            if "skin" in key:
                segment["color"] = (255, 255, 255)
            else:
                # stop to add other segmentation masks
                break
        try:
            for name_id in range(len(segment["name"])):
                if not skin and "skin" in key:
                    break
                if not nose and "nose" in key:
                    break
                if not glasses and "glasses" in key:
                    break
                if not eyes and "eye" in key:
                    break
                if not brows and "brows" in key:
                    break
                if not ears and "ear" in key:
                    break
                if not mouth and "mouth" in key:
                    break
                if not u_lip and "u_lip" in key:
                    break
                if not l_lip and "l_lip" in key:
                    break
                if not hair and "hair" in key:
                    # we want to remove the regions of masks that are below the hair mask
                    pass
                if not neck and "neck" in key:
                    break
                if not misc and "misc" in key:
                    break
                # open segmentation mask
                img = Image.open(data_path / sub_folder / (pic_id + segment["name"][name_id]))
                # recolor white to respective color of segment
                img = recolor_to_rgba(img, (255, 255, 255), segment["color"])
                # make black transparent
                img = make_color_transparent(img, (0, 0, 0))
                # save individual frames for later modification
                frames.append(img)
                # warp image
                if warp:
                    img = add_random_warp(img, seed=seed)
                    warped_frames.append(img)
        except FileNotFoundError:
            unprocessed_segments = unprocessed_segments + 1
            if unprocessed_segments == len(segments):
                print("No segmentation maps for file " + pic_id + " found! Stopping.")
                return
    # render frames
    if warp:
        result = render_frames(warped_frames)
    else:
        result = render_frames(frames)
    if showGUI:
        result = render_gui(frames)

    if not hair:
        # remove hair region again, this ensures no mask sticks out "below" the hair
        result = make_color_transparent(result, segments['hair']["color"])

    return im_resize(result)


def no_action(val):
    pass


def render_gui(frames):
    """
    Renders an opencv gui where you can warp and move segmentation masks and generate a new image based on that mask
    """
    try:
        _set_trackbars(0, 0, 0, 0, 0, 0)
    except cv2.error:
        pass

    cv2.namedWindow('Mask')
    cv2.moveWindow("Mask", 0, 20)
    cv2.createTrackbar('Select Mask', 'Mask', 0, len(frames) - 1, lambda x: _set_trackbars(x, 0, 0, 0, 0, 0))
    cv2.createTrackbar('Dilate', 'Mask', 0, 20, no_action)
    cv2.createTrackbar('Erode', 'Mask', 0, 20, no_action)
    cv2.createTrackbar('Rotate', 'Mask', -180, 180, no_action)
    cv2.createTrackbar('c', 'Mask', -100, 100, no_action)
    cv2.createTrackbar('f', 'Mask', -100, 100, no_action)

    cv2.setTrackbarMin('c', 'Mask', -100)
    cv2.setTrackbarMin('f', 'Mask', -100)
    cv2.setTrackbarMin('Rotate', 'Mask', -180)

    warped_frames = frames.copy()
    # reset settings
    settings = settings_old = [[0 for _ in range(5)] for _ in range(len(frames))]
    index_old = -1

    while 1:
        # get mask
        index = cv2.getTrackbarPos('Select Mask', 'Mask')

        # load settings
        if index is not index_old:
            dilate, erode, deg, c, f = settings[index]
            _set_trackbars(index, dilate, erode, deg, c, f)

        # get settings for mask
        dilate = cv2.getTrackbarPos('Dilate', 'Mask')
        erode = cv2.getTrackbarPos('Erode', 'Mask')
        deg = cv2.getTrackbarPos('Rotate', 'Mask')
        c = cv2.getTrackbarPos('c', 'Mask')
        f = cv2.getTrackbarPos('f', 'Mask')

        # save settings
        settings[index] = [dilate, erode, deg, c, f]

        # apply settings
        if settings == settings_old:
            img = add_warp(frames[index].copy(), dilate=dilate, erode=erode, deg=deg, c=c, f=f)
            warped_frames[index] = img

        result = render_frames(warped_frames)

        # show result
        mask = np.array(result.convert('RGB'))
        mask = mask[:, :, ::-1]

        cv2.imshow('Mask', mask)
        index_old = index
        settings_old = settings

        # escape key
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cv2.destroyWindow('Mask')

    return result


def _set_trackbars(index, dilate, erode, deg, c, f):
    """
    Helper function for the gui
    """
    cv2.setTrackbarPos('Select Mask', 'Mask', index)
    cv2.setTrackbarPos('Dilate', 'Mask', dilate)
    cv2.setTrackbarPos('Erode', 'Mask', erode)
    cv2.setTrackbarPos('Rotate', 'Mask', deg)
    cv2.setTrackbarPos('c', 'Mask', c)
    cv2.setTrackbarPos('f', 'Mask', f)


@click.command()
@click.option('--data_path', default='/path/to/CelebAMask-HQ/CelebA-HQ-img', help='Data root path.')
@click.option('--processed_path', default='/path/to/folder/', help='processed folder path')
@click.option('--train', default=0.8, help='% train split')
@click.option('--data_points', default=30000, help='number of data points you want to process')
@click.option('--save_a_b', '-ab', is_flag=True, help="Use if you want to save each image separately")
@click.option('--overlay_mask', is_flag=True, default=False, help="Overlay Mask on background.")
@click.option('--no_misc', is_flag=True, default=False, help="Do not include this mask.")
@click.option('--no_hair', is_flag=True, default=False, help="Do not include this mask.")
@click.option('--no_neck', is_flag=True, default=False, help="Do not include this mask.")
@click.option('--no_ears', is_flag=True, default=False, help="Do not include this mask.")
def preprocess_images(data_points: int, data_path: str, processed_path: str, train: float,
                      save_a_b: bool, no_misc: bool, no_hair: bool, no_neck: bool, no_ears: bool,
                      overlay_mask: bool):
    """
    Preprocess images according to pix2pix requirements
    """
    print(f'Processing with following options:\n'
          f'Overlay Mask: {overlay_mask}, save_a_b: {save_a_b}, train-percentage: {train * 100}%\n'
          f'Misc: {not no_misc}, Hair: {not no_hair}, Neck: {not no_neck}, Ears {not no_ears}')
    data_path = Path(data_path)
    processed_path = Path(processed_path)

    if save_a_b:
        create_folder_structure(processed_path / 'A')
        create_folder_structure(processed_path / 'B')
    else:
        create_folder_structure(processed_path)

    test = 1 - train
    for i in tqdm(range(data_points)):
        # load and resize 512x512 px mask image
        mask_img = create_mask_from_id(i,
                                       skin=True, nose=True, glasses=True, eyes=True, brows=True, ears=not no_ears,
                                       mouth=True, u_lip=True, l_lip=True,
                                       hair=not no_hair, neck=not no_neck, misc=not no_misc,
                                       )
        # load and resize corresponding 1028x1028px image
        normal_img = im_resize(Image.open(data_path / f"{i}.jpg"))

        # split dataset
        if i < test * data_points:
            folder = "test"
        elif i < (test + train * train) * data_points:
            folder = "train"
        else:
            # using the val images as train as well as we dont really have validation with pix2pix
            folder = "train"

        if overlay_mask:
            mask_img = make_color_transparent(mask_img, (0, 0, 0))
            mask_background = normal_img.copy()
            mask_background.paste(mask_img, (0, 0), mask_img)
            mask_background.convert("RGB")
            mask_img = mask_background

        # save images
        if save_a_b:
            normal_img.save(processed_path / "A" / folder / f"{i}.png", format='PNG', subsampling=0, quality=100)
            mask_img.save(processed_path / "B" / folder / f"{i}.png", format='PNG', subsampling=0, quality=100)
        else:
            # merge the two images into one 256x512px image
            output = merge_images_side_by_side(im_resize(normal_img), im_resize(mask_img))
            # output.save(processed_path / folder / f"{i}.png", format='PNG', subsampling=0, quality=100)
            output.save(processed_path / folder / f"{i}.png", format='PNG', subsampling=0, quality=100)


if __name__ == '__main__':
    preprocess_images()
