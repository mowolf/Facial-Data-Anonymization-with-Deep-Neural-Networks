from pathlib import Path

import cv2 as cv
from PIL import Image
from tqdm import tqdm
import click
from preprocess.utils import make_color_transparent, get_files, merge_images_side_by_side, overlay_two_images
from preprocess.enhance_masks import dilate


def cutout_face(img_path: str, mask_path: str, processed_path: str, face_mask_path: str):
    """Cuts out face of person
    :param face_mask_path:
    :param processed_path:
    :param img_path: path of folder containing images
    :param mask_path: path of folder containing masks
    """

    # get all filenames in the folders
    img_file_list = get_files(img_path, '.jpg')
    mask_file_list = get_files(mask_path, '.png')
    if img_file_list is None or mask_file_list is None:
        print("No image files found. Exiting")
        return
    if not len(img_file_list) == len(mask_file_list):
        print("Not the same amount og image files and mask files. Exiting")
        return

    # create folder for processed data
    processed_path = Path(processed_path)
    img_path = Path(img_path)
    face_mask_path = Path(face_mask_path)
    mask_path = Path(mask_path)
    processed_path.mkdir(exist_ok=True)

    for file in tqdm(img_file_list):
        # Load images
        face_img = Image.open(img_path / file.name)
        face_mask_img = Image.open(face_mask_path / f"{file.stem}.png")
        mask_img = Image.open(mask_path / f"{file.stem}.png")

        # Create dilated face mask as transparent, background is black
        face_mask_img_dilated = cv.imread(cv.samples.findFile(str(face_mask_path / f"{file.stem}.png")))
        face_mask_img_dilated = dilate(face_mask_img_dilated, 20)
        face_mask_img_dilated = cv.cvtColor(face_mask_img_dilated, cv.COLOR_BGR2RGB)
        face_mask_img_dilated = Image.fromarray(face_mask_img_dilated)
        face_mask_img_dilated = make_color_transparent(face_mask_img_dilated, (255, 255, 255))

        # Make face mask transparent, background is black
        face_mask_img = make_color_transparent(face_mask_img, (255, 255, 255))

        # copy face mask onto the mask image
        overlaid_mask = overlay_two_images(mask_img, face_mask_img)
        # Make background transparent
        overlaid_mask = make_color_transparent(overlaid_mask, (0, 0, 0))

        # overlay dilated face onto image
        overlaid_img = overlay_two_images(face_img, face_mask_img_dilated)
        # overlay mask onto image
        overlaid_mask = overlay_two_images(overlaid_img, overlaid_mask)

        # save result
        merge_images_side_by_side(overlaid_mask, overlaid_img).save(processed_path / file.name, format='PNG',
                                                                    subsampling=0, quality=100)

@click.command()
@click.option('--img_path', default='/home/mo/datasets/solo/original/test', help='Img  path.')
@click.option('--mask_path', default='/home/mo/datasets/solo/masks/test', help='Mask  path.')
@click.option('--face_mask_path', default='/home/mo/datasets/solo/face_only/test', help='Face only mask  path.')
@click.option('--processed_path', default='/home/mo/datasets/cutout_face_dilated/test', help='Save path')
def _cutout_face(img_path: str, mask_path: str, processed_path: str, face_mask_path: str):
    cutout_face(img_path, mask_path, processed_path, face_mask_path)


if __name__ == '__main__':
    _cutout_face()
