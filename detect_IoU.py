"""
python detect_IoU.py kind-donkey-84 data/temporalDataset_video.mp4 --show-crop --device cuda:0
"""
# %%

# Simulate command-line arguments
import sys
#sys.argv = ['ipykernel_launcher.py', 'kind-donkey-84', 'data/temporalDataset_video.mp4', '--end',  '120', '--show-crop', '--device', 'cuda:1'] # zum testen nur 120 sekunden
sys.argv = ['ipykernel_launcher.py', 'kind-donkey-84', 'data/temporalDataset_video.mp4', '--show-crop', '--device', 'cuda:1']
# python detect_IoU.py kind-donkey-84 data/temporalDataset_video.mp4 --end 120 --show-crop --device cuda:1

import argparse
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from src.utils.common import set_seeds, set_worker_seeds, simple_logger, split_dataset, split_dataset_by_sequence, split_dataset_by_sequence_from_lists
from src.utils.interface import Detector
from src.utils.visualization import draw_egopath, drawAnnotation
import matplotlib.pyplot as plt

# for evaluating on the dataset video
from src.utils.evaluate import compute_iou
"""Computes the Intersection over Union (IoU) between two binary masks.

    Args:
        input (numpy.ndarray or PIL.Image.Image): Input mask. Can be a numpy array (True/False, 0/1, 0/255) or a PIL image.
        target (numpy.ndarray or PIL.Image.Image): Ground truth mask. Can be a numpy array (True/False, 0/1, 0/255) or a PIL image.

    Returns:
        float: The IoU score.
"""


from src.utils.postprocessing import rails_to_mask
"""Creates a binary mask of the detected region from the ego-path points.

    Args:
        rails (list): List containing the left and right rails lists of rails point coordinates (x, y).
        mask_shape (tuple): Shape (W, H) of the target mask.

    Returns:
        PIL.Image.Image: Binary mask of the detected region.

"""

import yaml
import json
import re

lösche_ersten_warmup_indices = 0    # 0, 9, 29
plotIoUs = True                     # um IoUs zu plotten

def lösche_warmup_indices(liste, löschen=lösche_ersten_warmup_indices, schritt=76):
    """
    löscht die ersten paar indices aus einer sequence herraus um einen fairen Vergleich zeischen single-fram-based model und LSTM model zu garantieren.
    jede GT ist damit gleich.
    """
    ergebnis = []
    for i in range(0, len(liste), schritt):
        ergebnis.extend(liste[i+löschen:i+schritt])
    return ergebnis

def extract_frame_indices(image_names):
    frame_indices = []
    
    # Muster für das Erkennen des Frame-Index
    pattern = r'frame_(\d+)\.png'
    
    for image_name in image_names:
        match = re.search(pattern, image_name)
        if match:
            # Den gefundenen Frame-Index zur Liste hinzufügen
            frame_indices.append(int(match.group(1)))
    
    return frame_indices

def get_sequence_starts(indices, warmup):
    sequence_starts = []
    
    if indices:
        sequence_starts.append(indices[0])
    
    # find the first indizes of the sequences
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            sequence_starts.append(indices[i])
    
    for i in range(len(sequence_starts)):
        if sequence_starts[i] - warmup <= 0:
            sequence_starts[i] = 0
        else:
            sequence_starts[i] = sequence_starts[i] - warmup

    return sequence_starts

def generate_rails_mask(shape, annotation):
    rails_mask = Image.new("L", shape, 0)
    draw = ImageDraw.Draw(rails_mask)
    rails = [np.array(annotation["left_rail"]), np.array(annotation["right_rail"])]
    for rail in rails:
        draw.line([tuple(xy) for xy in rail], fill=1, width=1)
    rails_mask = np.array(rails_mask)
    rails_mask[: max(rails[0][:, 1].min(), rails[1][:, 1].min()), :] = 0
    for row_idx in np.where(np.sum(rails_mask, axis=1) > 2)[0]:
        rails_mask[row_idx, np.nonzero(rails_mask[row_idx, :])[0][1:-1]] = 0
    return rails_mask

def generate_target_segmentation(rails_mask):
    target = np.zeros_like(rails_mask, dtype=np.uint8)
    row_indices, col_indices = np.nonzero(rails_mask)
    range_rows = np.arange(row_indices.min(), row_indices.max() + 1)
    for row in reversed(range_rows):
        rails_points = col_indices[row_indices == row]
        if len(rails_points) != 2:
            break
        target[row, rails_points[0] : rails_points[1] + 1] = 255
    return Image.fromarray(target)

def split_into_parts(ious, part_size=76):
    parts = [ious[i:i + part_size] for i in range(0, len(ious), part_size)]
    mean_values = [np.mean(part, axis=0) for part in parts]  # axis=0, um den Mittelwert über die Zeilen zu berechnen
    return parts, mean_values

# Funktion zum Erstellen des Plots für einen Teil der IOU-Werte
def plot_iou(iou_parts, sequence_num, mean_per_seq):
    plt.figure()
    plt.plot(iou_parts, marker='o')
    plt.title(f'IOU über Sequenz: {sequence_num} mit mean IoU: {mean_per_seq}')
    plt.xlabel('Index')
    plt.ylabel('IOU')
    plt.grid(True)
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Ego-Path Detection Inference Script")
    parser.add_argument(
        "model",
        type=str,
        help="Name of the trained model to use (e.g., 'chromatic-laughter-5').",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input file (image or video).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the destination directory for the output file. If not specified, the output is saved in the same directory as the input file.",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default="auto",
        help="Coordinates to use for cropping the input image or video ('auto' for automatic cropping, 'x_left,y_top,x_right,y_bottom' inclusive absolute coordinates for manual cropping, or 'none' to disable cropping).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Inference starting point in the input video in seconds. If not specified, starts from the beginning.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Inference ending point in the input video in seconds. If not specified, processes the video until the end.",
    )
    parser.add_argument(
        "--show-crop",
        action="store_true",
        help="If enabled, displays the crop boundaries in the visual output.",
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
    logger = simple_logger(__name__, "info")
    base_path = os.path.dirname(__file__)
    # Parse crop coordinates
    if args.crop == "auto":
        crop_coords = "auto"
    elif args.crop == "none":
        crop_coords = None
    else:
        crop_coords = tuple(map(int, args.crop.split(",")))

    detector = Detector(
        model_path=os.path.join(base_path, "weights", args.model),
        crop_coords=crop_coords,
        runtime="pytorch",
        device=args.device,
    )

    extension = os.path.splitext(args.input)[1]
    outname = f"{os.path.splitext(os.path.basename(args.input))[0]}_out{extension}"
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, outname)
    else:
        output_path = os.path.join(os.path.dirname(args.input), outname)

    logger.info(f"\nDetecting ego-path in {args.input}...")

    if extension in [".jpg", ".jpeg", ".png"]:
        frame = Image.open(args.input)
        for i in range(50 if crop_coords == "auto" else 1):
            crop = detector.get_crop_coords() if args.show_crop else None
            res = detector.detect(frame)
        draw_egopath(frame, res, crop_coords=crop).save(output_path)

    elif extension in [".mp4", ".avi"]:
        cap = cv2.VideoCapture(args.input)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        start_frame = int(args.start * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        max_frames = (
            int(args.end * fps) - start_frame if args.end is not None else total_frames
        )
        current_frame = 0
        logger.info(
            f"\nFRAMES: {total_frames}"
            + f" | RESOLUTION: {frame_width}x{frame_height}"
            + f" | FPS: {fps}"
        )
        progress_bar = simple_logger(f"{__name__}_progress", "info", terminator="\r")
        
        # ----------------------------------------------- frame list -----------------------------------------------

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
        #proportions = (0, 0, 38) # test on whole dataset
        #proportions = (30, 4, 4) # test only on test dataset
        #train_indices, val_indices, test_indices = split_dataset_by_sequence(images_path, proportions)

        train_sequence_indices = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 27, 29, 30, 31, 33, 34, 36, 37]
        val_sequence_indices = [19, 25, 26, 32]
        test_sequence_indices = [35, 3, 24, 28]
        # rausgenommen: 10

        # whole dataset in testset
        #test_sequence_indices = train_sequence_indices + val_sequence_indices + test_sequence_indices

        print(train_sequence_indices)
        print(val_sequence_indices)
        print(test_sequence_indices)

        train_indices, val_indices, test_indices = split_dataset_by_sequence_from_lists(images_path, train_sequence_indices, val_sequence_indices, test_sequence_indices)
        #print("test_indices:")
        #print(test_indices)
        #print("length of test indices: ", len(test_indices))
        #print("first train index: ", test_indices[0])
        #print("last train index: ", test_indices[-1])
        set_seeds(config["seed"])  # reset random state

        test_indices = lösche_warmup_indices(test_indices)
        
        #print(test_indices)
        #print("length of test indices: ", len(test_indices))

        with open(annotations_path) as json_file:
            annotations = json.load(json_file)
        imgs = [sorted(annotations.keys())[i] for i in test_indices]
        
        #print(imgs)
        #print("length of imgs: ", len(imgs))

        frame_list = extract_frame_indices(imgs)
        print(frame_list)

        start_indices = get_sequence_starts(frame_list, 500)
        print(start_indices)

        # Leere Liste für die Tupel
        matching_tuples = []

        # Iteriere durch die Frame-Indizes
        for index in frame_list:
            # Konvertiere den Index zu einem String und überprüfe, ob er in einem der Dateinamen enthalten ist
            index_str = f"{index:06d}"  # Formatiert die Zahl als String mit mindestens 6 Stellen (z.B. '018499')

            # Suche den Dateinamen, der diesen Frame-Index enthält
            for file_name in imgs:
                if index_str in file_name:
                    matching_tuples.append((index, file_name))  # Füge das Tupel (Dateiname, Frame-Index) hinzu
                    break  # Wenn der passende Dateiname gefunden wurde, breche die Schleife ab
                
        # Zeige die Liste der Tupel an
        #print(matching_tuples)
        #print(len(matching_tuples))

        frame_dict = dict(matching_tuples)

        # to do:
        # - richtige annotation heraus nehmen ✔️
        # - iou von diesen frames ausrechnen
            # auf der detection seite:
                # -> rails_to_mask von evaluate.py verwenden --> converts prediction to a binary mask ✔️
            # auf der ground trouth seite:
                # -> generate_rail_mask --> converts groundthrouth to a binary mask ✔️
                # -> generate_target_segmentation --> dann calculate iou ✔️
            # dann calculate IoU aus evaluate.py --> Computes the Intersection over Union (IoU) between two binary masks. ✔️

        ious = []

        while current_frame < max_frames:
            current_frame += 1
            if current_frame in start_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                for i in range(600):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    crop = detector.get_crop_coords() if args.show_crop else None
                    res = detector.detect(frame) # prediction
                    if current_frame in frame_list:
                        vis = draw_egopath(frame, res, opacity=0.5, color=(0, 189, 80), crop_coords=crop) # grün
                        pred = rails_to_mask(res, frame.size)                                  # converts prediction to a binary mask (pred)
                        img_name = frame_dict.get(current_frame, "Frame-Index nicht gefunden") # gets right annotation-key
                        annotation = annotations[img_name]                                     # gets annotation from json
                        vis = drawAnnotation(vis, annotation)                                  # draws annotations for visual comparison
                        rails_mask = generate_rails_mask(frame.size, annotation)               # converts GT to a binary mask --> only rails
                        target = generate_target_segmentation(rails_mask)                      # converts GT from only rails to a binary mask --> whole track-bed
                        ious.append(compute_iou(pred, target))                                 # computes IoU - between 2 binary masks (prediction and GT are rails and track-bed area)
                    else:
                        vis = draw_egopath(frame, res, crop_coords=crop) # blau
                    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
                    out.write(vis)
                    current_frame += 1
                    progress_bar.info(
                        f"Processed {current_frame:0{len(str(max_frames))}}/{max_frames} frames"
                        + f" ({current_frame/max_frames*100:.2f}%)"
                    )
        cap.release()
        out.release()
        logger.info("")

        test_iou = np.mean(ious).item()
        logger.info(f"Test IoU: {test_iou:.5f}")

        print(ious)
        print(len(ious))
        
        if plotIoUs:
            ious = np.array(ious) # convert to np array
            
            print("writing average IoUs to txt file ...")
            with open('calculateIoU_singleFrame_video_ious_kind-donkey-84_newDataset_ganzesDataset_0_frames_gelöscht.txt', 'w') as file:
                for item in ious:
                    file.write(f"{item}\n")  # Jeden Wert in einer neuen Zeile schreiben

            print("plotting IoUs of the Sequences ...")
            # IOU-Liste in 4 Teile aufteilen
            iou_parts, means_per_seq = split_into_parts(ious, 76-lösche_ersten_warmup_indices) # through one sequence (76-9 = 67)

            # Für jeden Teil einen separaten Plot erstellen
            for i, part in enumerate(iou_parts, start=1):
                plot_iou(part, i, means_per_seq[i-1])

    else:
        raise NotImplementedError

    logger.info(f"\nInference complete. Output saved to {output_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

# %%
