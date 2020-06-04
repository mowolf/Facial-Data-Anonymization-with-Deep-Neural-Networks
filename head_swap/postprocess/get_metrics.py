import json
from pathlib import Path

import click
import torch
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from preprocess.face_detector import get_face
from preprocess.utils import get_files, get_subfolders
from preprocess.facenet import facenet_embedding
from age_gender_model.age_gender import AgeGenderDetector

cosine_sim = torch.nn.CosineSimilarity(dim=0)


def save_load_bb(bb: tuple):
    try:
        (x, y, w, h) = bb
    except (ValueError, TypeError):
        (x, y, w, h) = bb[0]
    return x, y, w, h


def compute_iou(bb1: tuple, bb2: tuple) -> float:
    """
    Calculate IoU
    :param bb1: Bounding box of format x,y, w,h
    :param bb2:  Bounding box of format x,y, w,h
    :return: Intersection over Union
    """
    h1, w1, x1, y1 = save_load_bb(bb1)
    x2, y2, w2, h2 = save_load_bb(bb2)

    # determine the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    # compute areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = w1 * h1
    bb2_area = w2 * h2

    # compute the intersection over union
    union_area = float(bb1_area + bb2_area - intersection_area)
    if union_area > 0:
        iou = intersection_area / union_area
    else:
        iou = 0

    return iou


def _save_metrics(existing_metrics: dict, folder: str):
    """
    Saves metrics in metrics.txt
    :param existing_metrics: dict of all metrics
    :param folder: Destination to save item
    """
    # save metrics as json dump
    with open(folder / 'metrics.txt', 'w') as outfile:
        json.dump(existing_metrics, outfile)


def _load_metrics(folder: str) -> dict:
    """
    Loads metrics if existing in folder/metrics.txt
    :param folder: folder
    :return: metrics
    """
    try:
        with open(folder / 'metrics.txt', 'r') as f:
            existing_metrics = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        existing_metrics = {}
    return existing_metrics


def get_identity_loss(identity_1: torch.Tensor, identity_2: torch.Tensor, lambda_Dis: int = 1) -> float:
    """
    Calculates identity loss
    :param identity_1: embedding
    :param identity_2: embedding
    :param lambda_Dis: weighting of the loss
    :return: loss
    """
    # Similarity measure between x and y, where 0 means no similarity, and 1 means maximum similarity.
    cosine = cosine_sim(identity_1, identity_2)
    # change this to a
    loss = max(0, (1 - cosine.item()) * lambda_Dis)
    return loss


class MetricVisualizer:
    """
    Class for calculating and visualizing metrics in tensorboard.
     - use MetricVisualizer.log_to_tensorboard to log computed data to tensorboard
     -  Calculated metrics get saved in a .txt file in each respective folder
    """

    def __init__(self, file_type: str = '.png'):
        self.file_type = file_type
        self.loaded = False

    def load_model(self):
        self.ageGenderModel = AgeGenderDetector()

    def compute_or_load_metrics(self, path: Path, ground_truth_folder: str = 'ground_truth', ):
        """
        Computes or loads metrics if existing
        :param path: path to folder
        :param ground_truth_folder: path to ground truth
        """
        print("Calculation or loading of precomputed metrics.")

        self.path = Path(path)
        self.ground_truth_folder = ground_truth_folder

        metrics = self._calculate_metrics_for_subfolders()
        self.truth_metrics = metrics[self.path / self.ground_truth_folder]
        del metrics[self.path / self.ground_truth_folder]
        self.test_metrics = metrics

    def calc_metric_for_folder(self, folder: Path):
        """
        Calculates metrics for a folder
        :param folder: folder to calculate metrics
        :return: metrics of this folder
        """
        if not self.loaded:
            self.load_model()

        # load metrics if existing
        existing_metrics = _load_metrics(folder)
        new_metrics = False
        files = get_files(folder, file_type=self.file_type)

        # code to rename keys if needed
        # loaded_metrics = existing_metrics.copy()
        # for key, item in loaded_metrics.items():
        #     if "_fake_B" in key:
        #         key_new = key.replace('_fake_B', '')
        #         existing_metrics[key_new] = existing_metrics.pop(key)
        #         new_metrics = True

        for idx, file in enumerate(tqdm(files)):
            # check if file was already computed
            if existing_metrics is not None and file.name in existing_metrics:
                continue
            # compute metrics
            metric = self.get_metric_for_file(file)
            # update dict
            existing_metrics.update(metric)
            new_metrics = True
            if idx % 1000 is 0 and new_metrics:
                _save_metrics(existing_metrics, folder)

        # save metrics
        if new_metrics:
            _save_metrics(existing_metrics, folder)

        return existing_metrics

    def get_metric_for_file(self, file: Path):
        """
        Calculates metrics for a single image
        :param file: path to single image file
        :return: metric dict
        """
        if not self.loaded:
            self.load_model()
            self.loaded = True

        metric = {}
        file = Path(file)

        extracted_face = get_face(file, better_fit=False)
        if len(extracted_face) > 1:
            print(f"Two faces detected in {file}, only using first face!")

        img = Image.open(file)

        embedding = facenet_embedding(img)
        if embedding is None:
            embedding = torch.zeros([512], dtype=torch.float)
        try:
            dlib_keypoints = extracted_face["0_" + str(file.name)]['keypoints'].tolist()
            dlib_bbox = extracted_face["0_" + str(file.name)]['rect_cv'],
        except KeyError:
            dlib_bbox = [0, 0, 0, 0]
            dlib_keypoints = torch.zeros([68, 2], dtype=torch.int).tolist()

        try:
            face_img = extracted_face["0_" + str(file.name)]["face_img"]
            gender, age = self.ageGenderModel.get_age_gender_from_face(face_img)
            # male2female > 1 -> male, male2female < 1 -> female
            male2female = gender[0][1] / gender[0][0]
        except KeyError:
            male2female = -1
            age = -1

        metric[str(file.name)] = {
            "identity": embedding.tolist(),
            "dlib_bbox": dlib_bbox,
            "dlib_keypoints": dlib_keypoints,
            "male2female": str(male2female),
            "age": str(round(float(age), 1))
        }
        return metric

    def log_to_tensorboard(self,
                           log_dir: Path = "./logs"):
        """
        Loggs all metrics to tensorboard
        :param log_dir: dir to save tensorboard loggings
        """
        self.writer = SummaryWriter(log_dir=Path(log_dir) / self.path.stem)
        self.writer.add_text("Info/Details", f"Log for {self.path}.")

        tb_losses = {}

        for folder, items in self.mean_diffs.items():
            item_index = 0
            folder = Path(folder).stem
            sorted_metrics = {k: v for k, v in sorted(items.items(), key=lambda x: x[1]["identity_loss"])}
            loss_dict = {"keypoint_loss": [], "identity_loss": [],
                            "bbox_iou": []}
            for losses in sorted_metrics.values():
                for loss_name, loss in losses.items():
                    tb_losses[loss_name] = {}
                    new_loss = torch.Tensor([loss])
                    # if new_loss != 0 and new_loss < 1000:
                    # There are some completely wrong values at the same position of all losses?!
                    loss_dict[loss_name].append(new_loss)
                    tb_losses[loss_name][folder] = new_loss
                    self.writer.add_scalars(f"Evaluation/{loss_name}", tb_losses[loss_name], item_index)
                item_index += 1
            for loss_name in list(set(losses.keys())):
                log_dict = {folder: np.mean(loss_dict[loss_name])}
                # create a horizontal bar
                self.writer.add_scalars(f"Mean/{loss_name}", log_dict, 0)
                self.writer.add_scalars(f"Mean/{loss_name}", log_dict, 1)
                self.writer.add_text(f"Info/Details/{folder}", f"Mean {loss_name}: {log_dict[folder]} +- {np.std(loss_dict[loss_name])} "
                                                         f"; min: {np.min(loss_dict[loss_name])} ; max {np.max(loss_dict[loss_name])}")

        # tensorboard does not show last plotted point so we plot it again... ?!
        self.writer.add_scalars(f"Mean/{loss}", log_dict, 1)

    def calc_mean_metrics(self):
        """
        Calculates mean metrics for each file in folder respective to the ground_truth folder
        :return: mean_diffs dict {file: loss_dict, ...}
        """
        mean_diffs = {}

        for key, value in self.test_metrics.items():
            mean_diffs[str(key)] = self._calc_metric_differences(value)

        self.mean_diffs = mean_diffs

        return mean_diffs

    def _calc_metric_differences(self, metrics: dict):
        """
        Accepts to dict with the same keys (filenames) that are dict with each key including
        key: {
            "identity": x,
            "dlib_bbox": y,
            "dlib_keypoints": z,
            }
        :param metrics:
        :return loss/differnece of metrics
        """
        diffs = {}
        missing = 0
        for key in metrics.keys():
            try:
                a = np.asarray(metrics[key]["dlib_keypoints"])
                b = np.asarray(self.truth_metrics[key]["dlib_keypoints"])
                if not a.shape == b.shape:
                    # this is needed for old metrics with a wrong shape
                    a = 0
            except KeyError:
                print(key)
                missing += 1
                continue

            keypoint_loss = np.linalg.norm(a - b)
            bbox_iou = compute_iou(metrics[key]["dlib_bbox"], self.truth_metrics[key]["dlib_bbox"])
            identity_loss = get_identity_loss(torch.Tensor(metrics[key]["identity"]),
                                              torch.Tensor(self.truth_metrics[key]["identity"]))
            diffs[key] = {"keypoint_loss": keypoint_loss, "bbox_iou": bbox_iou, "identity_loss": identity_loss}

        if missing > 0:
            print(f"{missing} keys missing in ground truth folder")

        return diffs

    def _calculate_metrics_for_subfolders(self):
        """
        Calculates the mean metric scores for each subfolder
        """
        folders = get_subfolders(self.path)
        all_metrics = {}

        for folder in tqdm(folders):
            print(f"\n\n{folder}:")
            metrics = self.calc_metric_for_folder(folder)
            all_metrics[folder] = metrics

        return all_metrics


@click.command()
@click.option('--data_path',
              help='Data root path.')
@click.option('--file_type', default='.png',
              help='Data root path.')
@click.option('--ground_truth_folder', default='ground_truth',
              help='Data root path for ground_truth folder.')
def get_mean_metrics_for_folders(data_path: Path, file_type: str, ground_truth_folder: str):
    m = MetricVisualizer(file_type=file_type)
    m.compute_or_load_metrics(data_path, ground_truth_folder=ground_truth_folder)
    m.calc_mean_metrics()
    m.log_to_tensorboard()


if __name__ == '__main__':
    get_mean_metrics_for_folders()
