import numpy as np
import cv2

from preprocess.utils import get_files


def video_to_frames(path: str, destination: str):
    """
    :param destination: path to folder where frames will be saved without trailing /
    :param path: path of video file
    """
    video = cv2.VideoCapture(path)
    success, image = video.read()
    count = 0
    while success:
        cv2.imwrite(f"{destination}/{count}.jpg", image)
        success, image = video.read()
        count += 1


def sort_descending(list_in):
    """
    Sorts list of str descending on !ALL! numbers contained in string
    :type list_in: list of str
    """
    return list(map(str, list_in)).sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


def write_video(img_array, destination, name_out="out"):
    """
    Writes a .mp4 file from images
    :param name_out: name of video file
    :param destination: path to store video
    :param img_array: sorted frames
    """
    height, width, _ = img_array[0].shape
    writer = cv2.VideoWriter(f'{destination}/{name_out}.mp4',
                             cv2.VideoWriter_fourcc(*'FMP4'), 25, (width, height))

    for i in range(len(img_array)):
        writer.write(img_array[i])

    writer.release()


def img_list_from_files(files: str):
    """
    :param files: filenames of image files as str
    :return: array of images
    """
    img_array = []
    for file in files:
        img = cv2.imread(file)
        img_array.append(img)
    return img_array


def pix2pix_results_to_frames(img_array):
    """
    Converts the results of the pix2pix model into frames by merging real_A and real_B with fake_A or fake_B
    :param img_array: sorted array containing the images of the result folder of the pix2pix network
    :return: merged frames
    """
    frames = []

    for i in range(int(len(img_array)/3)):

        try:
            left = cv2.resize(img_array[i * 3], dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            right = cv2.resize(img_array[i * 3 + 2], dsize=(512, 512), interpolation=cv2.INTER_NEAREST)

            scale = 512/img_array[i * 3 + 1].shape[0]
            middle = cv2.resize(img_array[i * 3 + 1], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            frames.append(np.concatenate((left, middle, right), axis=1))

            frames.append(img_array[i * 3+1])
        except:
            print("Error")

    return frames


def get_id(f):
    """
    Sort function to sort pix2pix output
    :param f: file
    :return: score
    """
    name = f.split('/')[-1][-10:-4]

    score = int(''.join(filter(str.isdigit, f.split('/')[-1][2:]))) * 3 + 1
    if name == 'real_A':
        score += 1
    if name == 'real_B':
        score -= 1

    return score


def pix2pix_results_to_video(path, destination=".", name_out="out"):
    """
    Converts pix2pix results to a video
    :param name_out: name of output video
    :param destination: folder to save video
    :param path: str path of folder containing the result images
    """
    files = list(map(str, get_files(path, '.png')))

    files.sort(key=get_id)

    img_array = img_list_from_files(files)
    frames = pix2pix_results_to_frames(img_array)
    write_video(frames, destination, name_out)


if __name__ == '__main__':
    print("No defined action.")
    # read_mp4('/home/mo/experiments/masterthesis/face_generation/preprocess/images/video.mp4')
    # files = get_files(path, '.jpg')
    # files = sort_descending(files)
    #
    # img_array = img_list_from_files(files)
    # write_video(img_array)

