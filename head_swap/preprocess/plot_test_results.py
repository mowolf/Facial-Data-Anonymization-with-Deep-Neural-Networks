from pathlib import Path

import click
import numpy as np
import cv2
from tqdm import tqdm

from preprocess.utils import get_subfolders, get_files


@click.command()
@click.option('--data_path',
              default='/home/mo/datasets/face_mask/train',
              help='Data root path.')
@click.option('--save_path',
              default='/home/mo/experiments/masterthesis/face_generation/evaluation/ffhq_ff_deeplab/',
              help='Figure save path.')
@click.option('--from_subfolders', is_flag=True, default=False, help="Use subfolders")
@click.option('--wide3', is_flag=True, default=False, help="3xn image, else 1xn image")
@click.option('--max_figures_per_folder', default=1)
@click.option('--n', default=10)
def plot_test_results(data_path: str, save_path: str, max_figures_per_folder: int, from_subfolders: bool, wide3: bool,
                      n: int):
    data_path = Path(data_path)
    save_path = Path(save_path)
    num = 0

    if from_subfolders:
        folders = get_subfolders(data_path)
    else:
        folders = [data_path]

    file_arr = []
    for folder in folders:
        files = get_files(folder, file_type='.png')
        file_arr.append(files)

    file_arr = [item for sublist in file_arr for item in sublist]

    cnt = 0
    out = []

    for file in tqdm(files[::100]):
        plot_files = [x for x in file_arr if x.name == file.name]

        if len(plot_files) == len(folders):
            for i, plot_file in enumerate(plot_files):
                row = cv2.imread(str(plot_file), 1)
                if cnt == 0:
                    out.append(row)
                else:
                    out[i] = np.concatenate((out[i], row))
                if cnt >= n or file == files[::100][-1]:
                    print(f"Saved figure {i}")
                    cv2.imwrite(f"{save_path}/figure_{data_path.stem}_{plot_file.parent.stem}_{num}.png", out[i])
                    if i == len(plot_files) - 1:
                        out = []
                        cnt = 0
                        num += 1
            cnt += 1
        if num >= max_figures_per_folder:
            print(f"Done. Saved {max_figures_per_folder} figure(s) for each folder at {save_path}")
            break


        # for file in tqdm(files[::100]):
        #
        #     if wide3:
        #         i += 1
        #         if i == 1:
        #             a = cv2.imread(str(file), 1)
        #         elif i == 2:
        #             b = cv2.imread(str(file), 1)
        #         else:
        #             c = cv2.imread(str(file), 1)
        #             row = np.concatenate((a, b, c), axis=1)
        #             if cnt == 0:
        #                 out = row
        #             else:
        #                 out = np.concatenate((out, row))
        #             if cnt >= n:
        #                 cv2.imwrite(f"{save_path}/figure_{data_path.stem}_{folder.name}_{num}.png", out)
        #                 cnt = 0
        #                 num += 1
        #                 if num > max_figures_per_folder:
        #                     break
        #             else:
        #                 cnt += 1
        #             i = 0
        #     else:
        #         row = cv2.imread(str(file), 1)
        #         if cnt == 0:
        #             out = row
        #         else:
        #             out = np.concatenate((out, row))
        #         if cnt >= n:
        #             cv2.imwrite(f"{save_path}/figure_{data_path.stem}_{folder.name}_{num}.png", out)
        #             cnt = 0
        #             num += 1
        #             if num >= max_figures_per_folder:
        #                 print(f"Done. Saved {n} figures at {save_path}")
        #                 break
        #         else:
        #             cnt += 1


if __name__ == '__main__':
    plot_test_results()
