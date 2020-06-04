import cv2
from PIL import Image
import json
from tqdm import tqdm
from pathlib import Path

from network.data import create_dataset
from network.models import create_model
from network.options.pippeline_options import PipelineOptions
from network.util.util import tensor2im
from preprocess.enhance_masks import clean_mask
from preprocess.face_detector import reinsert_aligned_into_image
from preprocess.utils import make_color_transparent, overlay_two_images

raw_folder = "raw"
test_folder = "test"
mask_folder = "mask"
generated_folder = "generated_faces"
generated_reinserted_folder = "generated_reinserted"
blended_reinserted_folder = "blended_reinserted"
to_blend_folder = "to_blend"
blended_folder = "blended"


def get_folders():
    return raw_folder, test_folder, mask_folder, generated_folder, generated_reinserted_folder, to_blend_folder, blended_folder


def create_folder(root_folder, pix2pix=False):
    (Path(root_folder) / test_folder).mkdir(parents=True, exist_ok=True)
    (Path(root_folder) / mask_folder).mkdir(parents=True, exist_ok=True)
    (Path(root_folder) / generated_folder).mkdir(parents=True, exist_ok=True)
    (Path(root_folder) / generated_reinserted_folder).mkdir(parents=True, exist_ok=True)
    (Path(root_folder) / blended_reinserted_folder).mkdir(parents=True, exist_ok=True)
    (Path(root_folder) / to_blend_folder).mkdir(parents=True, exist_ok=True)
    (Path(root_folder) / blended_folder).mkdir(parents=True, exist_ok=True)

    if pix2pix:
        (Path(root_folder) / "solo").mkdir(parents=True, exist_ok=True)
        (Path(root_folder) / "merged").mkdir(parents=True, exist_ok=True)


def get_model(opt, name="face2mask_weighted_l1_mask_19k_vanilla"):
    opt.name = name
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()
    return model


def get_pipeline_opt():
    opt = PipelineOptions().parse()  # get test options
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling - comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.dataset_mode = "single"

    return opt


def resize_long_edge(img, max_length: int = 1000):
    """
    :type max_length: int
    :param img: cv2/np image
    :param max_length: int
    :return: cv2/np image
    """
    long_edge = max(img.shape[0], img.shape[1])
    if long_edge > max_length:
        scale = max_length / long_edge
        cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    return img


def run_segmentation_model(model, opt, root_folder, warp=False):
    print("\n---- Running Face2Mask Model ----\n")

    dataset = create_dataset(opt)
    for i, data in tqdm(enumerate(dataset)):
        if i >= opt.num_test:
            break

        # run model and get processed image
        model.set_input(data)
        model.test()

        # get mask
        visuals_mask = model.get_current_visuals()
        mask = tensor2im(visuals_mask['fake_B'])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = clean_mask(mask)
        mask = Image.fromarray(mask)
        mask = make_color_transparent(mask, (0, 0, 0), tolerance=50)

        # open original
        img_path = model.get_image_paths()[0]
        img_name = img_path.split("/")[-1][:-4]
        image = Image.open(img_path)

        # add background to mask
        overlayed_mask = overlay_two_images(image, mask)

        # save file
        overlayed_mask.save(f'{root_folder}/{img_name}.png')

    print("SAVED GENERATED MASKS\n")


def run_generation_model(model, opt, root_folder, smoothEdge=20, min_width=30, margin=0):
    print("\n---- Running Mask2Face Model ---- \n")

    with open(f'{root_folder}/{test_folder}/coordinates.txt', 'r') as f:
        face_extraction_data = json.load(f)

    # logging
    num_too_small = 0
    num_replaced = 0
    insert_data = {}

    dataset = create_dataset(opt)
    for i, data in tqdm(enumerate(dataset)):
        if i >= opt.num_test:
            break

        # test model
        model.set_input(data)
        model.test()

        # get visuals
        visuals_face = model.get_current_visuals()
        img_path = model.get_image_paths()[0]
        key_name = img_path.split('/')[-1][:-4]
        idx = key_name.index("_")
        raw_file_name = key_name[idx + 1:] + ".png"

        # Tensor to image
        generated_image = tensor2im(visuals_face['fake_B'])
        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)

        # get face bounding box
        try:
            x, y, w, h = face_extraction_data[key_name]["rect_cv"]
        except KeyError:
            # this handles files that were not deleted and are not longer in the test set
            print(f"{key_name} key not found in dict face_extraction_data! Skipping.")
            continue

        # only continue if width of face that got reconstructed is larger that 50 pixels
        if w > min_width:
            out = _get_maybe_modified_file(raw_file_name, root_folder)

            # save generated image
            cv2.imwrite(f"{root_folder}/{generated_folder}/{key_name}.png", generated_image)

            # load keypoint data
            keypoints = face_extraction_data[key_name]["keypoints"]
            alignment_params = face_extraction_data[key_name]["alignment_params"]

            out, capture, out_margin, capture_margin, reinsert_range = reinsert_aligned_into_image(generated_image, out,
                                                                                                   alignment_params, keypoints,
                                                                                                   smoothEdge=smoothEdge,
                                                                                                   margin=margin,
                                                                                                   clean_merge=True)

            insert_data[key_name] = {'reinsert_range': reinsert_range}

            cv2.imwrite(f"{root_folder}/{generated_reinserted_folder}/{raw_file_name}", out)
            cv2.imwrite(f"{root_folder}/{to_blend_folder}/{key_name}.png", out_margin)

            num_replaced += 1
        else:
            num_too_small += 1

    with open(f"{root_folder}/{generated_folder}/coordinates.txt", 'w') as outfile:
        json.dump(insert_data, outfile)

    print(f"In total {num_replaced} faces replaced. "
          f"{num_too_small} incedents of faces with width < {min_width} were NOT replaced. \n")

    print("SAVED GENERATED FACE\n")


def _get_maybe_modified_file(raw_file_name, root_folder):
    # try to load modified, if not exists, load raw
    out = cv2.imread(f'{root_folder}/{generated_reinserted_folder}/{raw_file_name}', 1)
    if out is None:
        out = cv2.imread(f'{root_folder}/{raw_folder}/{raw_file_name[:-4]}.png', 1)
    return out


def run_blending_model(model, opt, root_folder, result_folder, blend_back=False):
    print("\n---- Running Face2Face Bleding Model ---- \n")

    if blend_back:
        with open(f'{root_folder}/{generated_folder}/coordinates.txt', 'r') as f:
            reinsert_rect = json.load(f)
        Path(f"{root_folder}/{blended_reinserted_folder}/{result_folder}").mkdir(parents=True, exist_ok=True)

    dataset = create_dataset(opt)
    for i, data in tqdm(enumerate(dataset)):
        if i >= opt.num_test:
            break

        # test model
        model.set_input(data)
        model.test()

        # get visuals
        visuals_face = model.get_current_visuals()
        img_path = model.get_image_paths()[0]
        file_name = img_path.split('/')[-1]

        # Tensor to image
        generated_image = tensor2im(visuals_face['fake_B'])
        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)

        # get key and file name
        key_name = file_name[:-4]
        raw_file_name = "_".join(file_name.split("_")[1:])

        # get original image
        out = _get_maybe_modified_file(raw_file_name, root_folder)
        # save generated image
        cv2.imwrite(f"{root_folder}/{blended_folder}/{result_folder}/{file_name}", generated_image)

        if blend_back:
            # get face bounding box
            x, y, x_end, y_end = reinsert_rect[key_name]["reinsert_range"]
            shape = out.shape

            # reinsert blended image
            generated_image = cv2.resize(generated_image,
                                         (x_end - x + min(shape[1] - x_end, 0), y_end - y + min(shape[0] - y_end, 0)))
            out[y:y_end, x:x_end, :] = generated_image

            # save
            cv2.imwrite(f"{root_folder}/{blended_reinserted_folder}/{result_folder}/{file_name}", out)

    print("SAVED BLENDED FACE\n")
