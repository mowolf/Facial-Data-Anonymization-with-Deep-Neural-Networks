import pandas as pd
import torch

from PIL import Image

from network.util.util import tensor2im
from preprocess.utils import get_files
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for facenet")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def facenet_embedding(img):
    """
    :param img: PIL image
    :return: facnet embedding of img
    """
    img = mtcnn(img)
    if img is None:
        return None
    img_stack = torch.stack([img]).to(device)
    embedding_list = resnet(img_stack)  # .detach.cpu()

    return embedding_list[0]


def facenet_distance(img1, img2, percentage=False, embedding=False):
    """
    Outputs distance between 0 and 2 with 0 is identical and 2 is most different
    :param embedding: bool - return embedding
    :param percentage: bool - return percentage
    :param img1: img1 tensor
    :param img2: img2 tensor
    :return: dist between 0 (identical) and 2 (most dissimilar), + embedding/percentange if True
    """
    img1 = mtcnn(Image.fromarray(tensor2im(img1)))
    img2 = mtcnn(Image.fromarray(tensor2im(img2)))

    if img1 is None or img2 is None:
        dist = 2
        embedding_list = [torch.zeros(512), torch.zeros(512)]
    else:
        img_stack = torch.stack([img1, img2]).to(device)
        embedding_list = resnet(img_stack)  # .detach().cpu()
        embedding_dist = embedding_list[0] - embedding_list[1]
        dist = embedding_dist.norm().item()

    if percentage:
        return dist, ((2 - dist) * 50)
    elif embedding:
        return dist, embedding_list
    else:
        return dist


if __name__ == '__main__':

    files = get_files("/home/mo/experiments/masterthesis/face_generation/preprocess/images/test/", ".jpeg")
    aligned = []
    names = []

    for f in files:
        image = Image.open(f)
        img_cropped = mtcnn(image)
        if img_cropped is not None:
            aligned.append(img_cropped)
            names.append("Mo")

    aligned = torch.stack(aligned)
    embeddings = resnet(aligned)

    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    print(pd.DataFrame(dists, columns=names, index=names))
