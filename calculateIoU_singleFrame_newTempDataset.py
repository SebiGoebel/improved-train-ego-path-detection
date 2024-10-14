#%%

"""
This skript calculates the IoU of a given model on the temporal test dataset.
The model has to ba a single-frame-based model.
The skript calculates the IoU on the test dataset of my own temporal dataset with the TEP annotations.

Command to start this script:
python calculateIoU_singleFrame.py <model> --device <cuda:GPU>
e.g.: python calculateIoU_singleFrame_newTempDataset.py kind-donkey-84 --device cuda:0

Example output of this script:

Evaluating single-frame-based model on temporal test dataset...
Test IoU: 0.97112

"""

# Simulate command-line arguments
import sys
sys.argv = ['ipykernel_launcher.py', 'kind-donkey-84', '--device', 'cuda:0']
# python calculateIoU_singleFrame_newTempDataset.py kind-donkey-84 --device cuda:0

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
from src.utils.common import set_seeds, set_worker_seeds, simple_logger, split_dataset, split_dataset_by_sequence, split_dataset_by_sequence_from_lists
from src.utils.dataset import PathsDataset
from src.utils.evaluate import IoUEvaluator

torch.use_deterministic_algorithms(True)

lösche_ersten_warmup_indices = True

def lösche_warmup_indices(liste, löschen=9, schritt=76):
    """
    löscht die ersten paar indices aus einer sequence herraus um einen fairen Vergleich zeischen single-fram-based model und LSTM model zu garantieren.
    jede GT ist damit gleich.
    """
    ergebnis = []
    for i in range(0, len(liste), schritt):
        ergebnis.extend(liste[i+löschen:i+schritt])
    return ergebnis

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

    # train_prop: 30  # number of the dataset to use for training
    # val_prop: 4  # number of the dataset to use for validation
    # test_prop: 4  # number of the dataset to use for testing
    images_path = "/srv/cdl-eml/datasets/temporalSwitchDataset_TEPForamt/images"  # path to the images directory
    annotations_path = "/srv/cdl-eml/datasets/temporalSwitchDataset_TEPForamt/labels/temporalLabels.json"  # path to the annotations file

    # Splitting temporal dataset
    set_seeds(config["seed"])  # set random state

    # sequence-lists:
    train_sequence_indices = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 27, 29, 30, 31, 33, 34, 36, 37]
    val_sequence_indices = [19, 25, 26, 32]
    test_sequence_indices = [35, 3, 24, 28]

    # whole dataset in testset
    #test_sequence_indices = train_sequence_indices + val_sequence_indices + test_sequence_indices

    print(train_sequence_indices)
    print(val_sequence_indices)
    print(test_sequence_indices)

    train_indices, val_indices, test_indices = split_dataset_by_sequence_from_lists(images_path, train_sequence_indices, val_sequence_indices, test_sequence_indices)
    print("test_indices:")
    print(test_indices)
    print("length of test indices: ", len(test_indices))
    print("first train index: ", test_indices[0])
    print("last train index: ", test_indices[-1])
    set_seeds(config["seed"])  # reset random state

    if lösche_ersten_warmup_indices:
        test_indices = lösche_warmup_indices(test_indices)
    
    print(test_indices)
    
    if len(test_indices) > 0:
        logger.info("\nEvaluating single-frame-based model on temporal test dataset...")
        test_dataset = PathsDataset(
            imgs_path=images_path,
            annotations_path=annotations_path,
            indices=test_indices,
            config=config,
            method="segmentation",
            img_crop="random",
            img_aug=False,
            img_rd_flip=True,
        )
        iou_evaluator = IoUEvaluator(
            dataset=test_dataset,
            model_path=model_path,
            crop = None,
            runtime="pytorch",
            device=device,
        )
        test_iou = iou_evaluator.evaluate()
        logger.info(f"Test IoU: {test_iou:.5f}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
