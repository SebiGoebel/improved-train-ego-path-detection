"""
CUDA_VISIBLE_DEVICES=0 python train_temporal.py regression efficientnet-b3 --device cuda:0
"""

# %%
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
from src.nn.lstm_model import RegressionNetCNN_LSTM_FC, RegressionNetCNN_FC_LSTM, RegressionNetCNN_FC_FCOUT, RegressionNetCNN_LSTM, RegressionNetCNN_LSTM_V2, RegressionNetCNN_LSTM_HEAD_V2, RegressionNetCNN_FLAT_FC, RegressionNetCNN_FC_FCOUT_V2
from src.utils.common import set_seeds, set_worker_seeds, simple_logger, split_dataset_by_sequence, split_dataset_by_sequence_from_lists
from src.utils.dataset_temporal import TemporalPathsDataset
from src.utils.sampler import TemporalSamplerIteratingSequenceSingleUsedImages, TemporalSamplerIteratingSequence, TemporalSamplerSingleSequence, TemporalSampler
from src.utils.interface import Detector
from src.utils.evaluate_temporal import IoUEvaluatorTemporal

from src.utils.trainer import train
from src.utils.visualization import draw_egopath
from src.utils.postprocessing import regression_to_rails, scale_rails
from PIL import Image

#torch.use_deterministic_algorithms(True)
import matplotlib.pyplot as plt
import numpy as np

# Simulate command-line arguments
import sys
#sys.argv = ['ipykernel_launcher.py', 'regression', 'efficientnet-b3', '--device', 'cuda:0']
# python train_temporal.py regression efficientnet-b3 --device cuda:0

# freesze layers
freeze_backbone = True
freeze_conv = True
# pool layer has no trainable parameters
freeze_pred_head = True # nur bei RegNetCNN_LSTM -> False

# ----- FUNCTION: copy layers -----

import torch.nn as nn
def copy_backbone_and_extra_layers(source_model, target_model, conv_layer_name_source, conv_layer_name_target, pool_layer_name_source, pool_layer_name_target, fc_layer_name_source, fc_layer_name_target, copy_fc):
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
    if copy_fc:
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

    train_dataset = TemporalPathsDataset(
        imgs_path=config["images_path"],
        annotations_path=config["annotations_path"],
        indices=train_indices,
        config=config,
        method=method,
        number_images_used=config["number_images_used"],
        img_crop="random",
        img_aug=False,
        img_rd_flip=False,
        to_tensor=True,
    )

    # ---------------------------------------------------- prints für dataset (dimensions and data) ----------------------------------------------------
    
    print("train_dataset: ", train_dataset)
    print("train dataset length: ", len(train_dataset))
    #print("train dataset shape: ", train_dataset[0])
    print("train dataset images: ", train_dataset[0][0].shape)
    print("train dataset labels traj: ", train_dataset[0][1].shape)
    print("train dataset labels ylim: ", train_dataset[0][2])
    print("-----------------------------------------------------------------")

    # train_dataset
    # len() = 2280
    # ein eintrag ist ein tuple aus (images, traj, ylim)
    # images = tensor mit [10, 3, 512, 512]
    # traj   = tensor mit [2, 64]
    # ylim   = tensor mit [1] (ein Eintrag)

    # Berechne die Anzahl der Zeilen und Spalten für das Rasterlayout
    sequence_tensor = train_dataset[0][0]
    n_images = sequence_tensor.shape[0] # 10

    n_cols = 8
    n_rows = (n_images + n_cols - 1) // n_cols

    # Erstelle eine Figur mit Subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    # Durchlaufe alle Bilder und zeige sie in den Subplots an
    for i in range(n_images):
        print("i: ", i)
        ax = axes[i // n_cols, i % n_cols]
        img = sequence_tensor[i].permute(1, 2, 0)  # Dimensionen von (3, 512, 512) zu (512, 512, 3) umordnen
        ax.imshow(img)
        ax.axis('off')  # Achsen ausblenden
        ax.set_title(f'Image {i + 1}', fontsize=8)

    # Bei weniger Subplots als Bilder, leere Subplots ausblenden
    for j in range(n_images, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')

    plt.tight_layout()
    plt.show()
    val_dataset = (
        TemporalPathsDataset(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=val_indices,
            config=config,
            method=method,
            number_images_used=config["number_images_used"],
            img_crop="random",
            img_aug=False,
            img_rd_flip=False,
            to_tensor=True,
        )
        if len(val_indices) > 0
        else None
    )
    
    # ---------------------------------------------------- SAMPLERS ----------------------------------------------------

    # ========= TRAIN - SAMPLER =========

    #temporal_train_sampler = TemporalSamplerIteratingSequence (              # 8 different sequences
    temporal_train_sampler = TemporalSamplerIteratingSequenceSingleUsedImages ( # 1 sequence with different data augementations
        train_dataset,
        batch_size=config["batch_size"], # 1
        sequence_length=config["sequence_length"],
        num_images=n_images, # 10
    )

    # ========= VAL - SAMPLER =========

    #temporal_val_sampler = TemporalSamplerIteratingSequence (            # 8 different sequences
    temporal_val_sampler = TemporalSamplerIteratingSequenceSingleUsedImages ( # 1 sequence with different data augementations
        val_dataset,
        batch_size=config["batch_size"], # 1
        sequence_length=config["sequence_length"],
        num_images=n_images, # 10
    )

    # ---------------------------------------------------- DATALOADERS ----------------------------------------------------

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        # idee:
        # lstm-input-tensor:        [batch_size, seq_len, inputsize (channel_size of feature map)]
        # needed tensor dimensions: [batch_size * seq_length, ]
        batch_size=config["batch_size"], # batch-size default: 1
        shuffle=False,                   # shuffle default: False
        num_workers=config["workers"],
        pin_memory=True,
        #drop_last=True, # letzte batch falls unvollständig wird gedropped
        worker_init_fn=set_worker_seeds,
        generator=torch.Generator().manual_seed(config["seed"]),
        sampler=temporal_train_sampler,
    )

    """

    print("train loader: ", train_loader)
    print("train loader length: ", len(train_loader)) # 1980 -> weil 30 sequences * 66 cut_outs * 1 batch_size
    # Erstelle einen Iterator aus dem DataLoader
    train_loader_iterator = iter(train_loader)
    train_features1, train_labels1, trains_ylimit1 = next(train_loader_iterator)
    train_features2, train_labels2, trains_ylimit2 = next(train_loader_iterator)
    #train_features, train_labels, trains_ylimit = next(iter(train_loader))
    print(f"Feature batch shape: {train_features2.size()}")
    print(f"Labels batch shape: {train_labels2.size()}")
    print("train labels: ", train_labels2)
    print(f"Labels batch shape ylimit: {trains_ylimit2}")
    
    # ---------------------------------------------------- showing batch of sequences + Groundtrouth  ----------------------------------------------------

    # Berechne die Anzahl der Zeilen und Spalten für das Rasterlayout
    n_batches, n_images_per_batch, _, _, _ = train_features2.shape
    n_cols = n_batches                                                             # Anzahl der Spalten für jedes Batch (8)
    n_rows = n_images_per_batch + 1                                                # Anzahl der Reihen (eine für jedes Batch) (76 + 1) -> +1 für groundtruth
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5)) # Erstelle eine Figur mit Subplots

    # falls rows oder cols 1 ist würde man eine 1D-Liste erhalten
    # um 2D-Liste zu garantieren
    if n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    # Durchlaufe alle Batches und Bilder und zeige sie in den Subplots an
    for batch in range(n_batches):
        for i in range(n_images_per_batch):
            ax = axes[i, batch]
            img = train_features2[batch, i].permute(1, 2, 0).numpy()  # Dimensionen von (3, 512, 512) zu (512, 512, 3) umordnen
            ax.imshow(img)
            ax.axis('off')  # Achsen ausblenden
            ax.set_title(f'Batch {batch + 1}, Img {i + 1}', fontsize=8)
    
    # Zeige die letzte Reihe (lestzten Bilder aus einer Sequence) doppelt an
    last_row_images = train_features2[:, -1]

    for i in range(n_batches):
        ax = axes[-1, i]  # Die letzte Zeile in jedem Batch
        img = last_row_images[i].permute(1, 2, 0).numpy()  # Dimensionen von (3, 512, 512) zu (512, 512, 3) umordnen
        # groundtruth
        img_uint8 = (img * 255).astype(np.uint8)
        image_pil = Image.fromarray(img_uint8)
        ground_truth_image_shape = image_pil.size
        train_labels_nparray = train_labels2[i].numpy()    # making numpy arrays out of labels
        trains_ylimits_nparray = trains_ylimit2[i].numpy() # making numpy arrays out of labels
        rails = regression_to_rails(train_labels_nparray, trains_ylimits_nparray)
        rails = scale_rails(rails, None, ground_truth_image_shape)
        rails = np.round(rails).astype(int)
        rails = rails.tolist()
        image = draw_egopath(image_pil, rails, crop_coords=None)
        ax.imshow(image)
        ax.axis('off')  # Achsen ausblenden
        ax.set_title(f'Groundtruth {i + 1}', fontsize=8)

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------- showing Groundtrouth of last image of first sequence ----------------------------------------------------

    # Create a 512x512 white image
    white_image = np.ones((512, 512, 3), dtype=np.uint8) * 0
    white_image_pil = Image.fromarray(white_image)
    ground_truth_image_shape = white_image_pil.size
    print("image.size: ", white_image_pil.size)

    train_labels_nparray = train_labels2[0].numpy()
    trains_ylimits_nparray = trains_ylimit2[0].numpy()

    print("train_labels_nparray: ", train_labels_nparray)
    print("trains_ylimits_nparray: ", trains_ylimits_nparray)

    rails = regression_to_rails(train_labels_nparray, trains_ylimits_nparray)   # func -> regression_to_rails()
    print("rails: ", rails)                                                     # print
    rails = scale_rails(rails, None, ground_truth_image_shape)                  # func -> scale_rails()
    print("rails after scale rails: ", rails)                                   # print
    rails = np.round(rails).astype(int)                                         # func -> round()
    print("rails after round: ", rails)                                         # print
    rails = rails.tolist()
    image = draw_egopath(white_image_pil, rails, crop_coords=None)
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    #print("train dataset images: ", train_dataset[0][0].shape)
    #print("train dataset labels traj: ", train_dataset[0][1].shape)
    #print("train dataset labels ylim: ", train_dataset[0][2])
    
    sys.exit()
    
    """

    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config["batch_size"], # batch-size default: 1
            num_workers=config["workers"],
            pin_memory=True,
            #drop_last=True, # letzte batch falls unvollständig wird gedropped
            worker_init_fn=set_worker_seeds,
            generator=torch.Generator().manual_seed(config["seed"]),
            sampler=temporal_val_sampler,
        )
        if val_dataset is not None
        else None
    )

    # ---------------------------------------------------- building RegressionNetCNN_FC_LSTM model ----------------------------------------------------

    if method == "regression":
        model = RegressionNetCNN_FC_FCOUT_V2( # RegressionNetCNN_FC_LSTM, RegressionNetCNN_LSTM_FC, RegressionNetCNN_FC_FCOUT, RegressionNetCNN_LSTM
            backbone=config["backbone"],
            input_shape=tuple(config["input_shape"]),
            anchors=config["anchors"],
            pool_channels=config["pool_channels"],
            fc_hidden_size=config["fc_hidden_size"],
            pretrained=config["pretrained"],
        ).to(device)
    elif method == "classification":
        print("ClassificationNet is not supported !!!")
        raise ValueError
    elif method == "segmentation":
        print("SegmentationNet is not supported !!!")
        raise ValueError
    else:
        raise ValueError

    # ---------------------------------------------------- copying backbone layers ----------------------------------------------------

    # loading pretrained model
    base_path = os.path.dirname(__file__)
    # 'toasty-haze-299' -> 21 anchors (=> 43)
    # 'decent-bee-298'  -> 32 anchors (=> 65)
    # 'kind-donkey-84'  -> 64 anchros (=> 129)
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
        fc_layer_name_target='fc',
        copy_fc = True,
    )

    #print(detector.model) # pretrained model with trained backbone
    #print(model)          # RegressionNetCNN_FC_LSTM model
    
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
    sys.exit()
    
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
    
    # ---------------------------------------------------- freeze backbone layers ----------------------------------------------------

    # freeze backbone
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    if freeze_conv:
        for param in model.conv.parameters():
            param.requires_grad = False

    # pool layer has no trainable parameters
    
    # freeze fully connected layers
    if freeze_pred_head:
        for param in model.fc.parameters():
            param.requires_grad = False
    
    #print(model) # temporal model
    
    """
    #check if freezing was corret
    counter_layers = 0
    counter_frozen_layers = 0
    counter_trainable_layers = 0
    for name, param in model.named_parameters():
        counter_layers += 1
        if not param.requires_grad:
            print(f"Frozen Layer: {name}")
            counter_frozen_layers += 1
        else:
            print(f"Trainable Layer: {name}")
            counter_trainable_layers += 1
    print("-------")
    print("counter_layers: ", counter_layers)
    print("counter_frozen_layers: ", counter_frozen_layers)
    print("counter_trainable_layers: ", counter_trainable_layers)
    sys.exit()
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

    # --- Abspeichern der feature maps ---
    # Abspeichern der feature map nach cnn
    # self.saved_features_cnn = [] # Liste um featuremaps zu speichern nach dem CNN -> 2D Tensoren mit [batch_size * seq_len (1*10), channels (2048)]
    # self.saved_featrues_fc = []  # Liste um outputs zu speichern nach FC-Layers   -> 2D Tensoren mit [batch_size * seq_len (1*10), output_size (129)]
    #features_to_save = {
    #    'saved_features_cnn': model.saved_features_cnn,
    #    'saved_featrues_fc': model.saved_featrues_fc,
    #}

    #torch.save(features_to_save, 'feature_maps.pt')
    #print("Feature maps wurden erfolgreich gespeichert")

    if len(test_indices) > 0:
        logger.info("\nEvaluating on test set...")
        test_dataset = TemporalPathsDataset(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=test_indices,
            config=config,
            method="segmentation",
            number_images_used=config["number_images_used"],
            img_crop="random",          # tuple (eval crops) or str ("random" -> random crop) or None (whole image)
            img_aug=False,              # alle data augmentations auf False gesetzt --> kein ColorJitter
            img_rd_flip=False,          # alle data augmentations auf False gesetzt --> keine random Flips
        )

        print("lenght of test_dataset: ", len(test_dataset))

        iou_evaluator = IoUEvaluatorTemporal(
            dataset=test_dataset,
            model_path=save_path, # LSTM model path
            crop=None,        # "auto" -> autocrop technique for test dataset (with 50 iterations) // None for when random crop is done in dataset class
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
