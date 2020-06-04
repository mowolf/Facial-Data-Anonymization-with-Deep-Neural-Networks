import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def flip_image(img):
    return np.asarray(Image.fromarray(img).transpose(Image.FLIP_LEFT_RIGHT))


def _add_noise(image, noise_type="random"):
    if noise_type == "random":
        noise_types = ["gauss", "s&p", "poisson", "speckle"]
        noise_type = noise_types[random.randint(0, 3)]

    if noise_type == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def adjust_color_balance(img, balance):
    """
    Adjusts color balance randomly
    :param img: cv2 image
    :param balance: amount to adjust balance
    :return:
    """
    out = np.zeros(img.shape, dtype=int)
    out[:, :, 0] = ((1 + 2 * balance) * img[:, :, 0] + (1 - balance) * img[:, :, 1] + (1 - balance) * img[:, :,
                                                                                                      2]) / 3
    out[:, :, 1] = ((1 + 2 * balance) * img[:, :, 1] + (1 - balance) * img[:, :, 0] + (1 - balance) * img[:, :,
                                                                                                      2]) / 3
    out[:, :, 2] = ((1 + 2 * balance) * img[:, :, 2] + (1 - balance) * img[:, :, 0] + (1 - balance) * img[:, :,
                                                                                                      1]) / 3
    return out


def apply_transforms(img, no_rotate=False, no_contrast=False, no_color=False, no_border=True):
    """
    Apply transformations to image
    :param img: pil IMAGE
    :return: transformed image
    """
    # add border
    if not no_border:
        if 1 > random.getrandbits(4):
            img = random_border(img)
    # convert to Image

    img = Image.fromarray(img)
    # rotate
    if not no_rotate:
        if 1 > random.getrandbits(1):
            img = random_rotate(img)
    # change contrast
    if not no_contrast:
        if 1 > random.getrandbits(4):
            img = random_contrast(img)
    # convert to cv2
    img = np.asarray(img)
    # change color
    if not no_color:
        if 1 > random.getrandbits(5):
            img = random_color_balance(img)
    # # add color hue
    # if 1 > random.getrandbits(5):
    #     img = random_color_hue(img)

    return img


def random_contrast(img):
    img = ImageEnhance.Contrast(img).enhance(random.randint(8, 12) / 10)
    return img


def random_border(img):
    top = random.randint(0, 3)
    bottom = random.randint(0, 3)
    left = random.randint(0, 3)
    right = random.randint(0, 3)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, color)
    return img


def random_color_balance(img):
    balance = random.randint(0, 40) / 100
    img = adjust_color_balance(img, balance)
    return img


def random_color_hue(img):
    (b, g, r) = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    alpha = 1 / 5 * random.randint(1, 3) + 0.1
    color = (b, g, r)
    mask = np.ones_like(img) * color
    img = cv2.addWeighted(img, alpha, mask, alpha, 0, dtype=cv2.CV_64F)
    return img


def random_rotate(img):
    angle = random.randint(-10, 10)
    img = img.rotate(angle)
    return img
