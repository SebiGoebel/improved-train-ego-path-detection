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
from src.nn.model import ClassificationNet, RegressionNet, SegmentationNet
from src.utils.common import set_seeds, set_worker_seeds, simple_logger, split_dataset_by_sequence
from src.utils.dataset import PathsDataset
from src.utils.interface import Detector
from src.utils.evaluate import IoUEvaluator
from src.utils.trainer import train

#torch.use_deterministic_algorithms(True)

# ----- FUNCTION: copy layers -----

import torch.nn as nn
def copy_backbone_and_extra_layers(source_model, target_model, conv_layer_name_source, conv_layer_name_target, pool_layer_name_source, pool_layer_name_target, fc_layer_name_source, fc_layer_name_target):
    """
    Kopiert die Parameter (Weights und Biases) aller Layer im Backbone, sowie eines Conv-Layers und eines Pool-Layers
    von einem Quellmodell zu einem Zielmodell.

    :param source_model: Das Modell, aus dem die Parameter kopiert werden sollen.
    :param target_model: Das Modell, in das die Parameter kopiert werden sollen.
    :param conv_layer_name_source: Der Name des Conv-Layers im Quellmodell.
    :param conv_layer_name_target: Der Name des Conv-Layers im Zielmodell.
    :param pool_layer_name_source: Der Name des Pool-Layers im Quellmodell.
    :param pool_layer_name_target: Der Name des Pool-Layers im Zielmodell.
    """
    
    # 1. Kopiere alle Layer im Backbone
    for layer_name, source_layer in dict(source_model.named_modules()).items():
        if 'backbone' in layer_name:  # Anpassen je nach Struktur des Modells
            target_layer_name = layer_name.replace('backbone', 'backbone')
            if target_layer_name in dict(target_model.named_modules()):
                target_layer = dict(target_model.named_modules())[target_layer_name]
                if isinstance(source_layer, nn.Module) and isinstance(target_layer, nn.Module):
                    target_layer.load_state_dict(source_layer.state_dict())
                else:
                    raise ValueError(f"Die Layer {layer_name} und {target_layer_name} sind nicht kompatibel oder existieren nicht.")
    
    # 2. Kopiere den spezifizierten Conv-Layer
    source_conv_layer = dict(source_model.named_modules())[conv_layer_name_source]
    target_conv_layer = dict(target_model.named_modules())[conv_layer_name_target]
    
    if isinstance(source_conv_layer, nn.Conv2d) and isinstance(target_conv_layer, nn.Conv2d):
        target_conv_layer.load_state_dict(source_conv_layer.state_dict())
    else:
        raise ValueError(f"Die Conv-Layer {conv_layer_name_source} und {conv_layer_name_target} sind nicht kompatibel oder existieren nicht.")
    
    # 3. Kopiere den spezifizierten Pool-Layer
    source_pool_layer = dict(source_model.named_modules())[pool_layer_name_source]
    target_pool_layer = dict(target_model.named_modules())[pool_layer_name_target]
    
    if isinstance(source_pool_layer, nn.Module) and isinstance(target_pool_layer, nn.Module):
        target_pool_layer.load_state_dict(source_pool_layer.state_dict())
    else:
        raise ValueError(f"Die Pool-Layer {pool_layer_name_source} und {pool_layer_name_target} sind nicht kompatibel oder existieren nicht.")
    
    # 4. Kopiere die spezifizierten FC-Layer
    source_fc_layer = dict(source_model.named_modules())[fc_layer_name_source]
    target_fc_layer = dict(target_model.named_modules())[fc_layer_name_target]
    
    if isinstance(source_fc_layer, nn.Module) and isinstance(target_fc_layer, nn.Module):
        target_fc_layer.load_state_dict(source_fc_layer.state_dict())
    else:
        raise ValueError(f"Die FC-Layer {fc_layer_name_source} und {fc_layer_name_target} sind nicht kompatibel oder existieren nicht.")

# ----- FUNCTION: parsing arguments -----

def parse_arguments():
    parser = argparse.ArgumentParser(description="Ego-Path Detection Training Script")
    parser.add_argument(
        "method",
        type=str,
        choices=["regression", "classification", "segmentation"],
        help="Method to use for the prediction head ('regression', 'classification' or 'segmentation').",
    )
    parser.add_argument(
        "backbone",
        type=str,
        choices=[f"resnet{x}" for x in [18, 34, 50]]
        + [f"efficientnet-b{x}" for x in [0, 1, 2, 3]]
        + [f"mobilenet-{x}" for x in ["small", "large"]]
        + [f"densenet{x}" for x in [121, 161, 169, 201]],
        help="Backbone to use (e.g., 'resnet18', 'efficientnet-b3', 'mobilenet-small', 'densenet121').",
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
    method = args.method
    device = torch.device(args.device)
    logger = simple_logger(__name__, "info")
    base_path = os.path.dirname(__file__)

    with open(os.path.join(base_path, "configs", "global_temporal.yaml")) as f:
        global_config = yaml.safe_load(f)
    with open(os.path.join(base_path, "configs", f"{method}.yaml")) as f:
        method_config = yaml.safe_load(f)
    config = {
        **global_config,
        **method_config,
        "method": method,
        "backbone": args.backbone,
    }

    # Splitting temporal dataset
    set_seeds(config["seed"])  # set random state
    image_path = config['images_path']
    proportions = (config["train_prop"], config["val_prop"], config["test_prop"])
    train_indices, val_indices, test_indices = split_dataset_by_sequence(image_path, proportions)
    #print("proportions: ")
    #print(config["train_prop"], ",", config["val_prop"], ",", config["test_prop"])
    #print("train_indices:")
    #print(train_indices)
    #print("first train index: ", train_indices[0])
    #print("last train index: ", train_indices[-1])
    set_seeds(config["seed"])  # reset random state

    train_dataset = PathsDataset(
        imgs_path=config["images_path"],
        annotations_path=config["annotations_path"],
        indices=train_indices,
        config=config,
        method=method,
        img_aug=True,
        to_tensor=True,
    )
    val_dataset = (
        PathsDataset(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=val_indices,
            config=config,
            method=method,
            img_aug=True,
            to_tensor=True,
        )
        if len(val_indices) > 0
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=True,
        worker_init_fn=set_worker_seeds,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    print("len(train_dataset): ", len(train_dataset))
    print("len(train_loader): ", len(train_loader))
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            num_workers=config["workers"],
            pin_memory=True,
            worker_init_fn=set_worker_seeds,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        if val_dataset is not None
        else None
    )

    if method == "regression":
        model = RegressionNet(
            backbone=config["backbone"],
            input_shape=tuple(config["input_shape"]),
            anchors=config["anchors"],
            pool_channels=config["pool_channels"],
            fc_hidden_size=config["fc_hidden_size"],
            pretrained=config["pretrained"],
        ).to(device)
    elif method == "classification":
        model = ClassificationNet(
            backbone=config["backbone"],
            input_shape=tuple(config["input_shape"]),
            anchors=config["anchors"],
            classes=config["classes"],
            pool_channels=config["pool_channels"],
            fc_hidden_size=config["fc_hidden_size"],
            pretrained=config["pretrained"],
        ).to(device)
    elif method == "segmentation":
        model = SegmentationNet(
            backbone=config["backbone"],
            decoder_channels=tuple(config["decoder_channels"]),
            pretrained=config["pretrained"],
        ).to(device)
    else:
        raise ValueError

    # ---------------------------------------------------- copying backbone layers ----------------------------------------------------

    # loading pretrained model
    base_path = os.path.dirname(__file__)
    pretrained_model_name = 'kind-donkey-84'
    detector = Detector(
        model_path=os.path.join(base_path, "weights", pretrained_model_name),
        crop_coords=None,
        runtime="pytorch",
        device=args.device,
    )

    copy_backbone_and_extra_layers(
        source_model=detector.model, 
        target_model=model, 
        conv_layer_name_source='conv', 
        conv_layer_name_target='conv', 
        pool_layer_name_source='pool', 
        pool_layer_name_target='pool',
        fc_layer_name_source='fc',
        fc_layer_name_target='fc'
    )

    #print(detector.model) # pretrained model with trained backbone
    #print(model)          # RegressionNetCNN_LSTM_FC model
    
    """
    # checking difference in layers after copying
    counter_layers = 0
    counter_no_difference_in_layers = 0
    counter_difference_in_layers = 0
    counter_non_existing_layers = 0
    for key in model.state_dict():
        counter_layers += 1
        if key in detector.model.state_dict():
            if torch.equal(detector.model.state_dict()[key], model.state_dict()[key]):
                print(f"matching: {key}")
                counter_no_difference_in_layers += 1
            else:
                print(f"NOT MATCHING: {key}")
                counter_difference_in_layers += 1
        else:
            print(f"Layer {key} does not exist in the source model. Skipping layer...")
            counter_non_existing_layers += 1
    print("---------")
    print("counter_layers: ", counter_layers)
    print("counter_no_difference_in_layers: ", counter_no_difference_in_layers)
    print("counter_difference_in_layers: ", counter_difference_in_layers)
    print("counter_non_existing_layers: ", counter_non_existing_layers)
    
    # calculating difference in layers after copying
    counter_difference_in_layers = 0
    counter_non_existing_layers = 0
    for key in model.state_dict():
        if key in detector.model.state_dict():
            difference = torch.sum(torch.abs(detector.model.state_dict()[key] - model.state_dict()[key]))
            print(f"Difference score: {difference.item()} for {key}")
            if difference != 0:
                counter_difference_in_layers += 1
        else:
            print(f"Layer {key} does not exist in the source model. Skipping layer...")
            counter_non_existing_layers += 1
    print("---------")
    print("counter_difference_in_layers: ", counter_difference_in_layers)
    print("counter_non_existing_layers: ", counter_non_existing_layers)
    """

    # ---------------------------------------------------- setting up systems for training (wandb, loss-function, optimizer [Adam], scheduler [OneCycleLR]) ----------------------------------------------------

    wandb.init(
        project="train-ego-path-detection",
        config=config,
        dir=os.path.join(base_path),
    )
    save_path = os.path.join(base_path, "weights", wandb.run.name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    if method == "regression":
        criterion = TrainEgoPathRegressionLoss(
            ylimit_loss_weight=config["ylimit_loss_weight"],
            perspective_weight_limit=train_dataset.get_perspective_weight_limit(
                percentile=config["perspective_weight_limit_percentile"],
                logger=logger,
            )
            if config["perspective_weight_limit_percentile"] is not None
            else None,
        )
        if config["perspective_weight_limit_percentile"] is not None:
            set_seeds(config["seed"])  # reset random state
    elif method == "classification":
        criterion = CrossEntropyLoss()
    elif method == "segmentation":
        criterion = BinaryDiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = (
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config["learning_rate"],
            total_steps=config["epochs"],
            pct_start=0.1,
            verbose=False,
        )
        if config["scheduler"] == "one_cycle"
        else None
    )

    logger.info(f"\nTraining {method} model for {config['epochs']} epochs...")
    train(
        epochs=config["epochs"],
        dataloaders=(train_loader, val_loader),
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=save_path,
        device=device,
        logger=logger,
        val_iterations=config["val_iterations"],
    )

    if len(test_indices) > 0:
        logger.info("\nEvaluating on test set...")
        test_dataset = PathsDataset(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=test_indices,
            config=config,
            method="segmentation",
        )
        iou_evaluator = IoUEvaluator(
            dataset=test_dataset,
            model_path=save_path,
            runtime="pytorch",
            device=device,
        )
        test_iou = iou_evaluator.evaluate()
        logger.info(f"Test IoU: {test_iou:.5f}")
        wandb.log({"test_iou": test_iou})


if __name__ == "__main__":
    wandb.login()
    args = parse_arguments()
    main(args)
