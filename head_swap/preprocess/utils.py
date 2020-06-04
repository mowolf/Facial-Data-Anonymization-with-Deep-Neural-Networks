import cv2
from PIL import Image
import numpy as np
from pathlib import Path


def compute_canny_edges(img, low_thr=0, high_thr=120, kernel=(3, 3)):
    """
    :param low_thr: lower threshold of canny
    :param high_thr: higher threshold of canny
    :param kernel: Gauß kernel size as a tuple, ege (3, 3)
    :param img: cv2 or numpy array of an image
    :return: edge image of same size as input
    """
    if img is []:
        return img
    img = cv2.GaussianBlur(img, kernel, 0)
    return cv2.Canny(img, low_thr, high_thr)


def get_nested_key_values(dictionary, key):
    """
    :param key: sub key of which to return the values as an array
    :param dictionary: dict of dicts
    :return: array of the values of key
    """
    values = []
    for item in dictionary.values():
        values.append(item[key])
    return values


def get_segments():
    """
    :return: Segments corresponding to the CelebA-HQ masks
    """
    segments = {'skin': {'name': ["_skin.png"], 'color': (204, 0, 0)},
                'nose': {'name': ["_nose.png"], 'color': (76, 153, 0)},
                'glasses': {'name': ["_eye_g.png"], 'color': (204, 204, 0)},
                'l_eye': {'name': ["_l_eye.png"], 'color': (51, 51, 255)},
                'r_eye': {'name': ["_r_eye.png"], 'color': (51, 51, 255)},  # if this has its own color: (204, 0, 204)
                'brows': {'name': ["_l_brow.png", "_r_brow.png"], 'color': (0, 255, 255)},
                'l_ear': {'name': ["_l_ear.png"], 'color': (102, 51, 0)},
                'r_ear': {'name': ["_r_ear.png"], 'color': (255, 0, 0)},
                'mouth': {'name': ["_mouth.png"], 'color': (102, 204, 0)},
                'u_lip': {'name': ["_u_lip.png"], 'color': (255, 255, 0)},
                'l_lip': {'name': ["_l_lip.png"], 'color': (0, 0, 153)},
                'hair': {'name': ["_hair.png"], 'color': (0, 0, 204)},
                'neck': {'name': ["_neck.png"], 'color': (255, 153, 51)},
                'misc': {'name': ["_hat.png", "_cloth.png", "ear_r.png"], 'color': (255, 51, 153)},
                }
    return segments


def create_folder_structure(root_dir: Path):
    """
    Creates test, train and val folder at root_dir
    :param root_dir: Path of folder
    """
    (root_dir / "test").mkdir(parents=True, exist_ok=True)
    (root_dir / "train").mkdir(exist_ok=True)
    (root_dir / "val").mkdir(exist_ok=True)


def recolor_to_rgba(img: Image, color: tuple, result_color: tuple):
    """Recolors color to result color, converts to rgba to keep transparency
    :param img: image
    :param color: rgb color tuple, color you want to replace
    :param result_color: rgb color tuple, color you want to use to repaint region
    """
    # handle data
    data = np.array(img.convert("RGBA"))
    recolor_color(color, data, result_color)

    return Image.fromarray(data).convert("RGBA")


def recolor_color(color, data, result_color):
    red, green, blue, alpha = data.T
    color_areas = (red == color[0]) & (green == color[1]) & (blue == color[2])
    # recolor
    data[..., :-1][color_areas.T] = result_color


def make_color_transparent(img, color: tuple, tolerance=0):
    """Recolors color to result color, converts to rgba to keep transparency
    :param tolerance:
    :param approx: includes close regions
    :param img: image
    :param color: rgb color tuple, color you want to make transparent
    """
    img = np.array(img.convert("RGBA"))

    color_area = get_area_with_color(img, color, tolerance)
    img[..., 3:][color_area] = 0

    return Image.fromarray(img).convert("RGBA")


def get_area_with_color(img, color, tolerance=0):
    """
    Returns mask with only color region
    :param img:
    :param color:
    :param tolerance:
    :return:
    """
    # remove alpha channel
    red, green, blue = img.T[0:3]

    mask_high = (red <= color[0] + tolerance) & \
                (green <= color[1] + tolerance) & \
                (blue <= color[2] + tolerance)
    mask_low = (color[0] - tolerance <= red) & \
               (color[1] - tolerance <= green) & \
               (color[2] - tolerance <= blue)
    color_area = mask_low & mask_high

    return color_area.T


def get_subfolders(data_path, verbose=False):
    """
    Returns all subfolders of data_path
    :param data_path: str or Parh
    :return:
    """
    p = Path(data_path).glob('./*/')
    folders = [x for x in p if not x.is_file()]
    folders.sort()

    if verbose:
        print(f"Found {len(folders)} folder(s).\n")
    return folders


def get_files(data_path, file_type):
    """Returns all .filetype filenames of data_path folder
    :param file_type: e.g. .jpg
    :param data_path: path of folder to search for images
    """
    p = Path(data_path).glob('**/*')
    files = [x for x in p if x.is_file() and str(x).endswith(file_type)]

    if len(files) == 0 or files is None:
        print(f"No {file_type} files found in location {data_path}\n")
        return

    # print(f"Found {len(files)} in {data_path}\n")

    return files


def merge_images_side_by_side(image1, image2):
    """Merge two images into one, displayed side by side
    :param image1: first image file
    :param image2: second image file
    :return: the merged image
    """
    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result


def overlay_two_images(img1: Image, img2: Image):
    """Overlays two images onto each other
    :param img1: image file 1
    :param img2: image file 2
    """
    img1 = img1.convert("RGBA")
    img2 = img2.convert("RGBA")

    (width1, height1) = img1.size
    (width2, height2) = img2.size
    assert width1 == width2 and height1 == height2

    background = Image.new("RGBA", (width1, height1), "BLACK")
    background.paste(img1, (0, 0), img1)
    background.paste(img2, (0, 0), img2)

    return background


def cutout_segmentation(image: Image, segmentation: Image):
    """Cuts segmentation1 out of image
    :param image: Source image
    :param segmentation: Image of segmentation mask, everything white in this mask is kept in the image
    """

    processed_mask = make_color_transparent(segmentation, (255, 255, 255))
    cut_out_image = overlay_two_images(image, processed_mask).convert("RGB")

    return cut_out_image
