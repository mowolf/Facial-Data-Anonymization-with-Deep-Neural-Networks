# Segmentation Model 

## Installation
Please install the packages via poetry or your favourite package manager. With poetry you just need to do:
```
 poetry install
```

## Usage

Please setup the path to your dataset in segmentation/core/data/dataloader/celebahq.py.

### Train
-----------------
```
poetry run python scripts/train.py
```

You can change the parameters with following arguments.
```
--model model 
--backbone backbone 
--lr lr 
--epochs epochs
```

### Evaluation
-----------------

```
poetry run python scripts/eval.py  --save_pred --save_path /result/save/path/
```

### Model

- [DeepLabv3](https://arxiv.org/abs/1706.05587)

### Dataset

Use the https://github.com/switchablenorms/CelebAMask-HQ dataset.


### Attribution

Code is based on: https://github.com/Tramac/awesome-semantic-segmentation-pytorch/