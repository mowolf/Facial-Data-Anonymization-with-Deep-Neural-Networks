import cv2 as cv
import click
from PIL import Image
from skimage.metrics import structural_similarity

from preprocess.utils import get_files, get_segments, get_area_with_color
import numpy as np


def get_mean_color(img_matrix):
    """
    Calculates mean color of image
    :param img_matrix: image
    :return: mean
    """
    color_list = []
    for row in img_matrix:
        for color in row:
            if not color.max == 0:
                color_list.append(color)

    if len(color_list) == 0:
        color_list.append((0, 0, 0))
    mean = tuple((np.mean(color_list, axis=0)).astype(int))
    return mean


def dilate(img, dilatation_size: int, dilatation_type=cv.MORPH_ELLIPSE):
    """
    Dilates an image with Ellipse
    :param dilatation_type: cv2 dilation type
    :param img: image cv2 style or np
    :param dilatation_size: size n, kernel: 2*n+1 (n = 2)
    :return: dilated image
    """
    # dilate by 2*n+1 (n = 2)
    element = cv.getStructuringElement(dilatation_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    return cv.dilate(img, element)


def get_frames(img):
    """
    Converts image into array of images of segmentations
    :param img: img in RGB color space
    :return: frames
    """
    segments = get_segments()
    frame_dict = {}
    for key, segment in segments.items():
        color = segment['color']
        frame = np.zeros(img.shape, np.uint8)
        color_area = get_area_with_color(img, color, tolerance=30)
        frame[..., :3][color_area] = color
        # Reconvert to BGR
        frame_dict[key] = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    return frame_dict


def optimize_frames(frame_dict: dict, warp=False):
    """
    Optimizes the different mask regions
    :param warp: if true, masks gets warped
    :param frame_dict: dict with "key": image
    :return: optimized image
    """
    from preprocess.process_celebA_masks_pix2pix import add_random_warp
    frames = []
    for key, frame in frame_dict.items():

        # add padding around the image to make sure border does not influence any morpholgical operations
        # ! Padding should be  >> Kernel Size
        padding = 120
        frame = cv.copyMakeBorder(frame, padding, padding, padding, padding, cv.BORDER_CONSTANT, None, [0, 0, 0])

        # Set parameters for different regions
        dilate_size = 0
        # kernel_size_open = (0, 0) # enable if some regions should not have an opening
        if 'eye' in key:
            kernel_size_open = (1, 1)
            kernel_size_close = (3, 3)
            dilate_size = 2
        elif 'nose' in key:
            kernel_size_open = (3, 3)
            kernel_size_close = (33, 33)
            dilate_size = 1
        elif 'skin' in key:
            kernel_size_open = (5, 5)
            kernel_size_close = (88, 88)
        elif 'hair' in key or 'neck' in key:
            kernel_size_open = (11, 11)
            kernel_size_close = (44, 44)
            dilate_size = 2
        elif 'glasses' in key:
            kernel_size_open = (3, 3)
            kernel_size_close = (3, 3)
        elif 'brows' in key:
            kernel_size_open = (3, 3)
            kernel_size_close = (3, 3)
            pass
        elif 'ear' in key:
            kernel_size_open = (3, 3)
            kernel_size_close = (3, 3)
        elif 'mouth' in key or 'lip' in key:
            kernel_size_open = (1, 1)
            kernel_size_close = (9, 9)
        elif 'misc' in key:
            kernel_size_open = (55, 55)
            kernel_size_close = (22, 22)
            dilate_size = 2
        else:
            print(f"{key} key not found in segmentation dict, will be excluded from mask.")
            continue
        # open
        if kernel_size_open[0] > 0:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size_open)
            frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
        # close
        if kernel_size_close[0] > 0:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size_close)
            frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel)
        # dilate
        if dilate_size > 0:
            frame = dilate(frame, dilate_size)
        # remove border again
        frame = frame[padding:-padding, padding:-padding, :]
        max_score = -1
        if warp and 'skin' not in key:
            for i in range(20):
                warped = add_random_warp(Image.fromarray(frame))
                warped = np.asarray(warped)
                # compute difference of masks
                (score, _) = structural_similarity(frame, warped,
                                                   full=True, multichannel=True)
                if score > max_score:
                    out = warped
            # Score is always greater than -1 so out is always referenced
            frame = out
        # append to frames
        frames.append(frame)

    return frames


def clean_mask(mask, optimize=True, warp=False):
    """
    Cleans masks
    :param warp: if mask should be warped
    :param optimize: optimize masks
    :param mask: mask image, cv2
    :return: cleaned mask image
    """
    mask = clean_colors(mask)
    from preprocess.process_celebA_masks_pix2pix import render_frames
    if optimize:
        frames_dict = get_frames(mask)
        frames = optimize_frames(frames_dict, warp=warp)
        mask = render_frames(frames)

    return mask


def clean_colors(img, tolerance=30):
    """
    Cleans some color artifacts
    :param tolerance: tolerance in which rgb value is still acceped as of color
    :param img: input image
    :return: img with cleaned colors
    """
    segments = get_segments()

    for _, segment in segments.items():
        color = segment['color']
        color_area = get_area_with_color(img, color, tolerance)
        # apply color only to rgb channels
        img[..., :3][color_area] = color

    return img


@click.command()
@click.option('--data_path', default='/home/mo/experiments/CelebAMask-HQ/generated_masks', help='Data root path.')
@click.option('--processed_path', default='/home/mo/experiments/CelebAMask-HQ/generated_masks/dilated/',
              help='Data path to save images.')
def enhance_masks(data_path: str, processed_path: str):
    """
    Enhances masks as outputted from pix2pix, EXPERIMENTAL
    :param processed_path:
    :param data_path: data path for masks
    :return:
    """
    file_list = get_files(data_path, ".png")
    for file in file_list:
        src = cv.imread(cv.samples.findFile(str(file)))
        dilated_img = dilate(src, 2)

        output = clean_colors(dilated_img)

        im_rgb = cv.cvtColor(output, cv.COLOR_BGR2RGB)
        Image.fromarray(im_rgb).save(processed_path + file.stem + '.jpg', format='JPEG', subsampling=0, quality=100)


if __name__ == '__main__':
    enhance_masks()
