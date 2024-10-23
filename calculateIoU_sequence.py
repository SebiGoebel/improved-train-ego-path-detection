#%%

"""
This skript calculates the IoU of a given temporal model.
The model has to ba a sequence-based model.
The skript calculates the IoU on the test dataset of my own temporal dataset with the TEP annotations.

Command to start this script:
python calculateIoU_sequence.py <model> --device <cuda:GPU>
e.g.: python calculateIoU_sequence.py stellar-shape-288 --device cuda:0

Example output of this script:

Evaluating sequence-based model on temporal test dataset...
Test IoU: 0.97112

"""

# Simulate command-line arguments
#import sys
#sys.argv = ['ipykernel_launcher.py', 'vivid-feather-328', '--device', 'cuda:0']
# python calculateIoU_sequence.py stellar-shape-288 --device cuda:0


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
from src.utils.common import set_seeds, set_worker_seeds, simple_logger, split_dataset_by_sequence, split_dataset_by_sequence_from_lists
from src.utils.dataset_temporal import TemporalPathsDataset
from src.utils.evaluate_temporal import IoUEvaluatorTemporal

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

    # Splitting temporal dataset
    set_seeds(config["seed"])  # set random state
    image_path = config['images_path']
    #proportions = (config["train_prop"], config["val_prop"], config["test_prop"])
    #train_indices, val_indices, test_indices = split_dataset_by_sequence(image_path, proportions)
    train_sequence_indices = config["train_indices"]
    val_sequence_indices = config["val_indices"]
    test_sequence_indices = config["test_indices"]

    print(train_sequence_indices)
    print(val_sequence_indices)
    print(test_sequence_indices)

    train_indices, val_indices, test_indices = split_dataset_by_sequence_from_lists(image_path, train_sequence_indices, val_sequence_indices, test_sequence_indices)

    #print("train_indices:")
    #print(train_indices)
    #print("val_indices:")
    #print(val_indices)
    #print("test_indices:")
    #print(test_indices)
    #print("first train index: ", train_indices[0])
    #print("last train index: ", train_indices[-1])
    set_seeds(config["seed"])  # reset random state

    if len(test_indices) > 0:
        logger.info("\nEvaluating sequence-based model on temporal test dataset...")
        test_dataset = TemporalPathsDataset(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=test_indices,
            config=config,
            method="segmentation",
            number_images_used=config["number_images_used"],
            img_crop="random",          # tuple (eval crops) or str ("random" -> random crop) or None (whole image)
            img_aug=False,          # alle data augmentations auf False gesetzt --> kein ColorJitter
            img_rd_flip=False,  	# alle data augmentations auf False gesetzt --> keine random Flips
        )
        iou_evaluator = IoUEvaluatorTemporal(
            dataset=test_dataset,
            model_path=model_path, # LSTM model path
            crop=None,           # "auto" -> autocrop technique for test dataset (with 50 iterations) // None -> for when random crop is done in dataset class
            runtime="pytorch",
            device=device,
        )
        test_iou = iou_evaluator.evaluate()
        logger.info(f"Test IoU: {test_iou:.5f}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
