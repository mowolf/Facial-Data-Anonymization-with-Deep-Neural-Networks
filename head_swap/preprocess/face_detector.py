import dlib
import click
import cv2
import numpy as np
import json

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from preprocess.enhance_masks import dilate
from preprocess.utils import get_files, merge_images_side_by_side, make_color_transparent, compute_canny_edges, \
    get_subfolders

# load predictor and detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./preprocess/pretrained_models/shape_predictor_68_face_landmarks.dat')


def save_cutout_with_margin(img_to_transform, margin, x, y, w, h):
    """
    Safely cut image and add margin
    :param img_to_transform:
    :param margin:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """

    if w > h:
        y = int(y - (w - h) / 2)
        h = w
    elif h > w:
        x = int(x - (h - w) / 2)
        w = h
    total_margin = int(margin * h)
    # check if we collide with an edge
    shape = img_to_transform.shape
    x_max, y_max = shape[0], shape[1]
    if total_margin > y:
        total_margin = y
    if total_margin > x:
        total_margin = x
    if (x_max - x - total_margin) < total_margin:
        total_margin = x_max - x - total_margin
    if (y_max - y - total_margin) < total_margin:
        total_margin = y_max - y - total_margin
    x_face_margin = total_margin
    y_face_margin = total_margin

    new_y = y - y_face_margin
    new_x = x - x_face_margin
    new_w = w + 2 * x_face_margin
    new_h = h + 2 * y_face_margin

    transformed_img = img_to_transform[new_y:new_y + new_h,
                      new_x:new_x + new_w, :]

    return transformed_img, (new_x, new_y, new_w, new_h)


def keypoints_to_mask(keypoints, img_shape: tuple, dilate_radius: int, color: tuple):
    """
    Creates an image by converting keypoints to a polygon in a color
    :param keypoints: keypoints
    :param img_shape: shape of image
    :param dilate_radius: radius to dilate mask
    :param color: rgb tuple
    :return:  BGR color image
    """
    # crate empty mask
    mask = np.zeros((img_shape[0], img_shape[1]))

    # Create color image of just plain color
    color_im = np.zeros(img_shape, np.uint8)
    color_im[:] = color

    # Fill convex Hull
    cv2.convexHull(mask, keypoints, 1)
    # dilate mask
    mask = dilate(mask, dilate_radius)
    # convert to bool
    mask = mask.astype(np.bool)

    # create output image
    out = np.zeros(img_shape, np.uint8)
    # apply boolean mask to only copy wanted regions
    out[mask] = color_im[mask]

    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def face_mask_from_keypoints(keypoints, img):
    """
    Converts keypoints to face mask
    :param keypoints: tuple of keypoints
    :param img: image
    :return: mask
    """
    # define mask regions
    upper_keypoints = keypoints[18:27].copy()
    # invert direction
    upper_keypoints = upper_keypoints[::-1]

    # add padding to eyebrows
    padded_upper_keypoints = apply_padding_in_x(upper_keypoints)

    # add padded keypoints to make face larger
    skin_keypoints = np.concatenate((keypoints[0:17], padded_upper_keypoints))
    nose_keypoints = np.concatenate((keypoints[27:28], keypoints[31:36]))

    keypoint_dict = {
        'skin': {
            'keypoints': skin_keypoints, 'color': (204, 0, 0)},
        'eyebrow_l': {
            'keypoints': keypoints[18:22], 'color': (0, 255, 255)},
        'eyebrow_r': {
            'keypoints': keypoints[22:27], 'color': (0, 255, 255)},
        'nose': {
            'keypoints': nose_keypoints, 'color': (76, 153, 0)},
        'eye_l': {
            'keypoints': keypoints[36:42], 'color': (51, 51, 255)},
        'eye_r': {
            'keypoints': keypoints[42:48], 'color': (51, 51, 255)},
        'lips': {
            'keypoints': keypoints[48:60], 'color': (255, 255, 0)},
        'mouth': {
            'keypoints': keypoints[60:69], 'color': (102, 204, 0)}
    }

    mask = []
    for key, value in keypoint_dict.items():
        dilate_radius = 1
        if 'eyebrow' in key:
            dilate_radius = 2
        mask.append(keypoints_to_mask(value['keypoints'], img.shape, dilate_radius, value['color']))

    for i in range(len(mask) - 1):
        if i is 0:
            back = Image.fromarray(mask[i]).convert("RGBA")

        front = Image.fromarray(mask[i + 1]).convert("RGBA")
        front = make_color_transparent(front, (0, 0, 0))
        # back is always assigned as i starts from 0
        back.paste(front, (0, 0), front)

    return np.array(back.convert("RGB"))


def apply_padding_in_x(keypoints, pad=15):
    """
    Adds padding to keypoint in x direction
    :param keypoints: dlib keypoints
    :param pad: amount of padding
    :return:
    """
    for i in range(len(keypoints)):
        if i > 4:
            x = -pad
        else:
            x = pad
        keypoints[i] = (keypoints[i][0] + x, keypoints[i][1] - pad)
    return keypoints


def rect_to_bb(rect):
    """
    convert bounding box predicted by dlib to the format (x, y, w, h)
    :param rect:
    :return: (x,y,w,h)
    """
    x = max(0, rect.left())
    y = max(0, rect.top())
    w = max(0, rect.right() - x)
    h = max(0, rect.bottom() - y)

    return x, y, w, h


def shape_to_np(shape, dtype="int"):
    """
    Converts shape to numpy
    :param shape: shape object
    :param dtype: np dtype
    :return:
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def get_face_keypoints(image, rect):
    """
    Get dlib predictions
    :param image: image
    :param rect: rect
    :return: shape
    """
    shape = predictor(image, rect)
    shape = shape_to_np(shape)
    return shape


def get_mask_from_points(img, keypoints, smooth=0, dilate_size=0):
    """
    Returns image with TRANSPARENT (ALPHA CHANNEL = 0) face region
    :param smooth: kernel size of gaussian blurring
    :param dilate_size: % of image length to dilate mask with
    :param img: Image
    :param keypoints: dlib keypoints
    :return:
    """
    # create empty 2D mask filled with 0s
    mask = np.zeros((img.shape[0], img.shape[1]))
    # calculate convex hull from keypoints
    convex_hull = cv2.convexHull(np.array(keypoints))
    # Fill convex hull with ones
    mask = cv2.fillConvexPoly(mask, convex_hull, 1)
    if dilate_size > 0:
        # get kernel size
        k_size = int((dilate_size / 100) * img.shape[0])
        k_size = k_size if (k_size % 2 == 1) else k_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size, k_size))
        # Use morphological operations
        mask = cv2.morphologyEx(mask, op=cv2.MORPH_DILATE, kernel=kernel)
    # smooth Mask if needed
    if smooth > 0:
        # as a gaussian blur can only make my region of interest smaller we need to dilate mask to find balance
        k_size = 2 * smooth + 1
        k_dilate = k_size // 4

        mask = dilate(mask, k_dilate)
        mask = cv2.GaussianBlur(mask, (k_size, k_size), 0)

    # create transparency mask
    mask = (1 - mask) * 255
    mask = mask.astype(np.uint8)

    return mask


def make_mask_transparent(img, mask):
    """
    Returns image with a new alpha channel of: alpha channel value =  mask
    :param img: Image
    :param mask: Mask
    :return: Image with alpha channel value = mask*255
    """
    # convert rgb to rgba
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    # use mask as alpha channel
    img[:, :, 3] = mask
    return img


def get_faces(data_path, file_type=".jpg", overlay_keypoints=False, overlay_mask=True, align=False,
              margin_multiplier=0.45, reduce_by = 0):
    """
    Detect faces, KEYPOINTS ARE ALIGNED ON ORIGINAL IMAGE!!
    :param overlay_mask:
    :param overlay_keypoints:
    :param align: align faces
    :param margin_multiplier: How much of additional image should be added to detected face region
    :param data_path: folder path to folder containing images
    :param file_type: image file type e.g. '.jpg'
    :return: dict containing cut out face, coordinates and the original image file name
    {'face_img': cropped_img,
    'rect': (x, y, w, h),
    'original_img': '1.jpg'}
    """
    file_list = get_files(data_path, file_type)
    if file_list is None:
        print("No Files found. Exiting.")
        exit()

    if reduce_by > 0:
        # reduce to get only every 20th image
        print(f"Reducing file list by {reduce_by}!")
        file_list = file_list[::reduce_by]

    print('\nDetecting Faces:')
    extracted_faces = {}
    for f in tqdm(file_list):
        extracted_face = get_face(f,
                                  overlay_keypoints=overlay_keypoints,
                                  overlay_mask=overlay_mask,
                                  align=align,
                                  margin_multiplier=margin_multiplier)
        extracted_faces.update(extracted_face)

    return extracted_faces


def get_face(f, overlay_keypoints=False, overlay_mask=False, align=False,
             margin_multiplier=0.45, better_fit=True):
    """
    Returns face data from file path
    :param f:
    :param overlay_keypoints:
    :param overlay_mask:
    :param align:
    :param margin_multiplier:
    :return:
        {
            'face_img': face_img,
            'rect_cv': (x, y, w, h),
            'keypoints': keypoints,
            'cutout_face_img': cutout_face_img,
            'aligned_face_img': aligned_face_img,
            'mask_img': mask_img,
            'alignment_params': alignment_params
        }
    """
    f = Path(f)
    extracted_faces = {}
    img = cv2.imread(str(f), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)

    for i, det in enumerate(dets):
        # get keypoints
        keypoints = get_face_keypoints(gray, det)

        # remove face
        cutout_face_img = []
        # if cutout_face:
        # TODO: add method to remove face from keypoints
        #       cutout_face_img = remove_face(img, keypoints)
        if overlay_keypoints:
            # add keypoints as circles to image
            for (x, y) in keypoints:
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        aligned_face_img = []
        alignment_params = ''
        if align:
            # align image
            # desiredLeftEye = (0.35,0,35) # close face only cutout
            # Workaround to get two differently scaled images
            # desiredLeftEye = (0.44, 0.44)  # wide head + background curout
            # desiredFaceShape = (275, 275)
            # aligned_face_img1, _, _, _, _ = align_face(img, keypoints,
            #                                                                               desiredLeftEye=desiredLeftEye,
            #                                                                               desiredFaceShape=desiredFaceShape)
            desiredLeftEye = (0.41, 0.41)  # wide head + background curout
            desiredFaceShape = (256, 256)
            aligned_face_img, eyesCenter, angle, scale, (w_affine, h_affine) = align_face(img, keypoints,
                                                                                          desiredLeftEye=desiredLeftEye,
                                                                                          desiredFaceShape=desiredFaceShape)
            alignment_params = {"eyesCenter": (int(eyesCenter[0]), int(eyesCenter[1])),
                                "angle": float(angle),
                                "scale": float(scale),
                                "shape": (w_affine, h_affine),
                                "desiredLeftEye": desiredLeftEye}
            assert w_affine == h_affine
            # img = reinsert_aligned_into_image(aligned_face_img, img, angle, desiredLeftEye, eyesCenter, scale,
            # w_affine)

        # get face image from bounding box
        if better_fit:
            (x, y, w, h) = calculate_tight_bounding_box(np.asarray(keypoints))
            face_img, new_bbox = save_cutout_with_margin(img, margin_multiplier, x, y, w, h)
            (x, y, w, h) = new_bbox
        else:
            (x, y, w, h) = rect_to_bb(det)
            face_img = get_face_from_bounding_box(img, margin_multiplier, (x, y, w, h))

        mask_img = []
        if overlay_mask:
            # create masks from keypoints
            mask_img = face_mask_from_keypoints(keypoints, img)
            mask_img = get_face_from_bounding_box(mask_img, margin_multiplier, (x, y, w, h))

        # save result in dict
        extracted_faces[f'{i}_{f.name}'] = {'face_img': face_img,
                                            'rect_cv': (x, y, w, h),
                                            'keypoints': keypoints,
                                            'cutout_face_img': cutout_face_img,
                                            'aligned_face_img': aligned_face_img,
                                            # 'aligned_face_img1': aligned_face_img1,
                                            'mask_img': mask_img,
                                            'alignment_params': alignment_params
                                            }
    return extracted_faces


def calculate_tight_bounding_box(keypoints):
    """
    Calculates tight fitting bbox to keypoints
    :param keypoints: x,y, xmax, y,max
    :return: new bbox, (x,y,w,h)
    """

    x_min, y_min = min(keypoints[:, 0]), min(keypoints[:, 1])
    x_max, y_max = max(keypoints[:, 0]), max(keypoints[:, 1])

    w_margin = int((x_max - x_min) * 0.15)
    x_max = x_max + w_margin
    x_min = x_min - w_margin

    y_min = y_min - int(0.7 * (y_max - y_min))
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)


def get_face_from_bounding_box(img, margin_multiplier, bounding_box):
    """
    Returns face image from bounding box
    :param img:
    :param margin_multiplier:
    :param bounding_box:
    :return:
    """
    (x, y, w, h) = bounding_box
    # convert to square
    if w < h:
        w = h
    else:
        h = w
    # add margin
    margin = round(h * margin_multiplier)
    # get image size
    y_max = img.shape[0]
    x_max = img.shape[1]
    # recast x,y,w,h
    y = min(max(0, y - margin), y_max)
    x = min(max(0, x - margin), x_max)
    w = min(min(x + 2 * margin + w, x_max) - x, min(y + 2 * margin + h, y_max) - y)
    h = w
    # get image of face only, crop with bounding box from dlib
    face_img = img[y:y + h, x:x + w, :].copy()

    return face_img


def reinsert_aligned_into_image(aligned_face_img, img, alignment_params,
                                keypoints, smoothEdge=20, margin=0, clean_merge=False):
    """
    Reinserts aligned image into original image by inverting the affine transformation
    :param margin: margin to add afterwards
    :param smoothEdge: adding a gaussian blurred edge
    :param alignment_params: dict with following content: angle, scale, shape, desiredLeftEye, eyesCenter
    :param keypoints: keypoint dict from dlib
    :param aligned_face_img: cv2 img
    :param img: cv2 img original background image
    :return: cv2 img, reinserted, of shape(img)
    """
    angle = float(alignment_params["angle"])
    scale = 1 / float(alignment_params["scale"])
    width = int(alignment_params["shape"][0])
    desiredLeftEye = [float(alignment_params["desiredLeftEye"][0]), float(alignment_params["desiredLeftEye"][1])]
    rotationPoint = [int(alignment_params["eyesCenter"][0]), int(alignment_params["eyesCenter"][1])]

    # save original image for later referenece
    original_img = img.copy()
    # get mask of face

    if clean_merge:
        # get keypoints of brows
        upper_keypoints = keypoints[18:27].copy()
        # add padding to eyebrows
        padded_upper_keypoints = apply_padding_in_x(upper_keypoints[::-1], pad=7)
        # add padded keypoints to make face larger
        keypoints = np.concatenate((keypoints[0:17], padded_upper_keypoints))
        # get mask that is smoothed at the edge
        mask = get_mask_from_points(img, keypoints, smooth=smoothEdge, dilate_size=5)
        # make mask transparent in original image
        removed_face_img = make_mask_transparent(img, mask)
        # convert to PIL
        removed_face_img = Image.fromarray(removed_face_img).convert("RGBA")

    # Reinsert face int original image
    # get inverse Matrix
    l_face = aligned_face_img.shape[0]
    m1 = int(l_face * 0.5)
    m2 = int(desiredLeftEye[0] * l_face)
    M = cv2.getRotationMatrix2D((m1, m2), -angle, scale)
    long_edge_size = width / abs(np.cos(np.deg2rad(angle)))
    w_original = int(scale * long_edge_size)
    h_original = int(scale * long_edge_size)
    # get offset
    tX = w_original * 0.5
    tY = h_original * desiredLeftEye[1]
    M[0, 2] += (tX - m1)
    M[1, 2] += (tY - m2)
    # rewarp aligned face
    rewarped_img = cv2.warpAffine(aligned_face_img, M, (w_original, h_original), flags=cv2.INTER_CUBIC)
    # Get Start positions of face in image
    x_start = rotationPoint[0] - round(0.5 * w_original)
    y_start = rotationPoint[1] - round(desiredLeftEye[0] * h_original)
    # clean artifacts on the edges
    rewarped_img = clean_edges(rewarped_img)
    # copy rewarped image into original image
    y_size = rewarped_img.shape[1]
    x_size = rewarped_img.shape[0]
    for k in range(y_size):
        for j in range(x_size):
            y = y_start + k
            x = x_start + j
            if not (rewarped_img[k, j, :] == [0, 0, 0]).all():
                try:
                    img[y, x, :] = rewarped_img[k, j, :]
                except IndexError:
                    # this happens at the edges of the image and can safely be ignored
                    pass

    # convert to PIL
    img = Image.fromarray(img)
    # Paste original image with cutout face onto rewarped image
    if clean_merge:
        img.paste(removed_face_img, (0, 0), removed_face_img)
    # else:
    # new = Image.fromarray(original_img.copy())
    # new.paste(img, (0, 0))
    # img = new

    # reconvert into Cv2 image type
    img = np.asarray(img)
    # Now we have the image with the pasted generated face at the correct position

    if clean_merge:
        # We want to blend this again -> copy over the face region with poisson blending to adjust colors
        # invert mask to get the mask of the face (and not the mask of the face)
        mask = 255 - mask
        # Find center of mask => Location of the center of the source image in the destination image.
        # we need to find center of the bounding box of the mask
        center = bounding_box_center(mask)
        # Normal Cloning: texture (gradient) of the source image is preserved in the cloned region. Mixed Cloning:
        # texture (gradient) of the cloned region is determined by a combination of the source and the destination
        # images. Mixed Cloning does not produce smooth regions because it picks the dominant texture ( gradient )
        # between the source and destination images.
        img_clone = cv2.seamlessClone(img, original_img, mask, center, cv2.NORMAL_CLONE)
    else:
        img_clone = img

    # remove the rest of the image but a small margin
    # make margin robust, check if we exceed bounds and adjust margin respectively
    margin_img_clone = img_clone.copy()
    margin_original_img = original_img.copy()
    y_size_original = margin_img_clone.shape[1]
    x_size_original = margin_img_clone.shape[0]

    x_start = 0 if x_start < 0 else x_start
    y_start = 0 if y_start < 0 else y_start
    # apply margin
    margin = max(
        min(
            y_start - max(0, y_start - margin),
            x_start - max(0, x_start - margin),
            min(y_start + y_size + margin, y_size_original) - y_start - y_size,
            min(x_start + x_size + margin, x_size_original) - x_start - x_size,
        ),
        0
    )
    margin_img_clone = margin_img_clone[y_start - margin:y_start + y_size + margin,
                       x_start - margin:x_start + x_size + margin, :]
    margin_original_img = margin_original_img[y_start - margin:y_start + y_size + margin,
                          x_start - margin:x_start + x_size + margin, :]
    reinsert_range = (x_start - margin, y_start - margin, x_start + x_size + margin, y_start + y_size + margin)

    return img_clone, original_img, margin_img_clone, margin_original_img, reinsert_range


def bounding_box_center(img):
    """
    Returns center coordinates
    :param img: cv2 image
    :return: center of image
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    center = (int((cmin + cmax) / 2), int((rmin + rmax) / 2))

    return center


def clean_edges(img, pixels_to_ommit=4):
    """
    This function removes the first #pixels_to_ommit pixels that are non black from every side of the image
    (top, bottom, left, right)
    :param img: image, cv2
    :param pixels_to_ommit: integer of how many pixels to remove
    :return: image
    """
    w, h, _ = img.shape

    for _k, _j in [[[0, w, 1], [0, h, 1]], [[0, h, 1], [0, w, 1]], [[w - 1, -1, -1], [h - 1, -1, -1]],
                   [[h - 1, -1, -1], [w - 1, -1, -1]]]:
        for k in range(_k[0], _k[1], _k[2]):
            passed_non_black_pixels = 0
            for j in range(_j[0], _j[1], _j[2]):
                if not (img[k, j, :] == [0, 0, 0]).all():
                    passed_non_black_pixels += 1
                    if passed_non_black_pixels > pixels_to_ommit:
                        break
                    else:
                        img[k, j, :] = [0, 0, 0]
    return img


def align_face(img, keypoints, desiredLeftEye=(0.35, 0.35), desiredFaceShape=(256, 256)):
    """
    Aligns a face so that left eye is at desiredLeftEye position
    adapted from https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    :param img: cv2 image, cut previously to just contain face
    :param keypoints: keypoints from dlib
    :param desiredLeftEye: position of left eye in aligned image
    :param desiredFaceShape: output image size
    :return: aligned face image, cv2
    """
    desiredFaceWidth = desiredFaceShape[0]
    desiredFaceHeight = desiredFaceShape[1]
    # get keypoints of the eyes
    leftEyePts = keypoints[36:42]
    rightEyePts = keypoints[42:48]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                  (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    aligned_face_img = cv2.warpAffine(img, M, (desiredFaceWidth, desiredFaceHeight), flags=cv2.INTER_CUBIC)

    return aligned_face_img, eyesCenter, angle, scale, (desiredFaceWidth, desiredFaceHeight)


def process_faces_for_pix2pix(data_path, file_type, processed_path, get_edges=False,
                              cutout_face=False, overlay_keypoints=False, overlay_mask=False,
                              align=False, AB=False, margin_multiplier=0.45):
    """
    Detects faces in all images in data_path, saves faces and the coordinates in processed_path as a dataset for pix2pix
    :param AB: merges images
    :param align: aligns face
    :param overlay_mask: creates mask from keypoints
    :param overlay_keypoints: overlays keypoints on image
    :param cutout_face: removes face from image
    :param get_edges: computes canny edges
    :param data_path: Path to folder containing images to be analysed
    :param file_type: image file type e.g. '.jpg'
    :param processed_path: Path to save detected faces and coordinate text file
    """
    processed_path = Path(processed_path)
    data_path = Path(data_path)
    data = {}

    faces = get_faces(data_path, file_type,
                      overlay_keypoints=overlay_keypoints,
                      overlay_mask=overlay_mask,
                      align=align,
                      margin_multiplier=margin_multiplier
                      )

    print('Processing faces:')
    for name, face in tqdm(faces.items()):
        # name = f"{data_path.stem}_{name.split(file_type)[0]}"
        name = f"{name.split(file_type)[0]}"
        try:
            face_img = cv2.cvtColor(face['face_img'], cv2.COLOR_RGB2BGR)
        except cv2.error:
            continue

        face_img = Image.fromarray(face_img).resize((256, 256), Image.NEAREST)
        data[name] = {'rect_cv': face['rect_cv'],
                      'keypoints': face['keypoints'].tolist(),
                      'alignment_params': face['alignment_params']}

        if get_edges:
            B_img = compute_canny_edges(cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR))
            B_img = Image.fromarray(B_img)
        if cutout_face:
            B_img = Image.fromarray(cv2.cvtColor(face['cutout_face_img'],
                                                 cv2.COLOR_RGB2BGR))
        if overlay_mask:
            face_img = Image.fromarray(cv2.cvtColor(face['mask_img'],
                                                    cv2.COLOR_RGB2BGR))
        if align:
            try:
                align_img = cv2.cvtColor(face['aligned_face_img'], cv2.COLOR_RGB2BGR)
                # align_img1 = cv2.cvtColor(face['aligned_face_img1'], cv2.COLOR_RGB2BGR)
            except cv2.error:
                continue
            face_img = Image.fromarray(align_img)
            # face_img1 = Image.fromarray(align_img1)
        if AB:
            out = merge_images_side_by_side(face_img, B_img)
        else:
            out = face_img

        out.save(processed_path / f"{name}.png", format='PNG', subsampling=0, quality=100)
        #
        # face_img1.save(processed_path / f"{name}_large.png", format='PNG', subsampling=0, quality=100)

    # save .json file with info about the rect
    with open(processed_path / f'coordinates.txt', 'w') as outfile:
        json.dump(data, outfile)

    print(f"Coordinate file and extracted faces were saved at {processed_path}")
    print(f"{data.__len__()} face(s) found.\n")

    if data == {}:
        return False
    else:
        return True


@click.command()
@click.option('--data_path', default='./preprocess/images/test',
              help='Data root path.')
@click.option('--save_path',
              default='./preprocess/test_prepared_images',
              help='Data processed path.')
@click.option('--file_type', default='.png', help='Data file type.')
@click.option('--overlay_keypoints', '-k', is_flag=True, default=False, help='Overlay keypoints on image.')
@click.option('--overlay_mask', '-m', is_flag=True, default=False, help='Overlay mask on image.')
@click.option('--cutout_face', '-g', is_flag=True, default=False, help='Remove Face.')
@click.option('--margin_multiplier', default=0.45, help='by what margin image should be extended')
def _process_faces_for_pix2pix(data_path: str,
                               save_path: str,
                               file_type: str,
                               overlay_keypoints: bool,
                               overlay_mask: bool,
                               cutout_face: bool,
                               margin_multiplier: int,
                               search_subfolders: bool = False):
    """
    Decorated process_faces function with @click.options
    :param data_path: Data root path.
    :param processed_path: Data processed path.
    :param file_type: Data file type.
    """
    if search_subfolders:
        folders = get_subfolders(data_path)
    else:
        folders = [data_path]

    save_path = Path(save_path)

    for f in tqdm(folders):
        process_faces_for_pix2pix(f, file_type, save_path,
                                  cutout_face=cutout_face,
                                  overlay_keypoints=overlay_keypoints,
                                  overlay_mask=overlay_mask,
                                  margin_multiplier=margin_multiplier,
                                  align=True)


if __name__ == '__main__':
    _process_faces_for_pix2pix()
