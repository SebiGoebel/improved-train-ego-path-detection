"""
This skript calculates the IoU of a given model.
The model has to ba a single-frame-based model.
The skript calculates the IoU on the test dataset of railsem19 with the TEP annotations.

Command to start this script:
python calculateIoU_singleFrame.py <model> --device <cuda:GPU>
e.g.: python calculateIoU_singleFrame.py kind-donkey-84 --device cuda:0

Example output of this script:

Evaluating single-frame-based model on TEP test dataset...
Test IoU: 0.97112

"""


import argparse
import json
import os
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import wandb
import yaml

from src.nn.loss import (
    BinaryDiceLoss,
    CrossEntropyLoss,
    TrainEgoPathRegressionLoss,
)
from src.utils.common import set_seeds, set_worker_seeds, simple_logger, split_dataset
from src.utils.dataset import PathsDataset
from src.utils.evaluate import IoUEvaluator

torch.use_deterministic_algorithms(True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Ego-Path Detection Inference Script")
    parser.add_argument(
        "model",
        type=str,
        help="Name of the trained model to use (e.g., 'chromatic-laughter-5').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"]
        + [f"cuda:{x}" for x in range(torch.cuda.device_count())],
        help="Device to use ('cpu', 'cuda', 'cuda:x' or 'mps').",
    )

    return parser.parse_args()

def main(args):
    device = torch.device(args.device)
    logger = simple_logger(__name__, "info")
    base_path = os.path.dirname(__file__)
    model_path=os.path.join(base_path, "weights", args.model)

    with open(os.path.join(model_path, "config.yaml")) as f:
        config = yaml.safe_load(f)
    config = {
        **config,
    }

    images_path = "/srv/cdl-eml/datasets/railsem19/rs19_val/jpgs/rs19_val"
    annotations_path = "/srv/cdl-eml/datasets/railsem19/egopath/rs19_egopath.json"

    set_seeds(config["seed"])  # set random state
    with open(annotations_path) as json_file:
        indices = list(range(len(json.load(json_file).keys())))
    random.shuffle(indices)
    proportions = (config["train_prop"], config["val_prop"], config["test_prop"])
    train_indices, val_indices, test_indices = split_dataset(indices, proportions)
    set_seeds(config["seed"])  # reset random state

    if len(test_indices) > 0:
        logger.info("\nEvaluating single-frame-based model on TEP test dataset...")
        test_dataset = PathsDataset(
            imgs_path=images_path,
            annotations_path=annotations_path,
            indices=test_indices,
            config=config,
            method="segmentation",
            img_crop="random",          # tuple (eval crops) or str ("random" -> random crop) or None (whole image)
            img_aug=False,          # alle data augmentations auf False gesetzt --> kein ColorJitter
            img_rd_flip=False,  	# alle data augmentations auf False gesetzt --> keine random Flips
        )
        iou_evaluator = IoUEvaluator(
            dataset=test_dataset,
            model_path=model_path,
            crop=None,           # "auto" -> autocrop technique for test dataset (with 50 iterations) // None -> for when random crop is done in dataset class
            runtime="pytorch",
            device=device,
        )
        test_iou = iou_evaluator.evaluate()
        logger.info(f"Test IoU: {test_iou:.5f}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
