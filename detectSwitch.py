"""
This skript saves the points of a single horizontal line and plots the behaviour on a graph.
"""

#%%
import argparse
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image

from src.utils.common import simple_logger
from src.utils.interface import Detector
from src.utils.visualization import draw_egopath

import plotly.graph_objects as go # for plots
import plotly.io as pio
from plotly.subplots import make_subplots # for subplots
pio.renderers.default = 'browser'  # Oder 'notebook' für statische Bilder im Jupyter Notebook
import yaml

from src.utils.autocrop import Autocropper


welcher_punkt = 0


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

    autocropper = Autocropper(detector.config)

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
        for _ in range(50 if crop_coords == "auto" else 1):
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
        
        plot_list_rail_left = []
        plot_list_rail_right = []
        plot_list_difference = []
        plot_list_min_x = []
        plot_list_max_x = []
        plot_list_auto_crop = []

        progress_bar = simple_logger(f"{__name__}_progress", "info", terminator="\r")
        while current_frame < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            crop = detector.get_crop_coords() if args.show_crop else None
            res = detector.detect(frame)

            rails_coords = autocropper.rails_coords(res)
            
            #print(rails_coords[0])
            #print(rails_coords[2])
            #print(res)
            #print(res[0][welcher_punkt][0]) # rail left
            #print(res[1][welcher_punkt][0]) # rail right
            #print(crop)

            point_left = [res[0][welcher_punkt][0], current_frame]  # rail left
            point_right = [res[1][welcher_punkt][0], current_frame] # rail right
            difference_points = [res[1][welcher_punkt][0]-res[0][welcher_punkt][0], current_frame] # difference
            point_min_x = [rails_coords[0], current_frame] # smallest rec (not crop cords)
            point_max_x = [rails_coords[2], current_frame] # smallest rec (not crop cords)
            if crop:
                auto_crops = [crop[0], crop[1], crop[2], current_frame]  # left, top, right, current_frame


            #print(point_left)
            #print(point_right)

            plot_list_rail_left.append(point_left)
            plot_list_rail_right.append(point_right)
            plot_list_difference.append(difference_points)
            plot_list_min_x.append(point_min_x)
            plot_list_max_x.append(point_max_x)
            if crop:
                plot_list_auto_crop.append(auto_crops) # left, top, right, current_frame

            vis = draw_egopath(frame, res, crop_coords=crop)
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

        #print(plot_list_rail_left)
        #print(plot_list_rail_right)

        # rail left x points
        x_coords_left = [point[0] for point in plot_list_rail_left]
        y_coords_left = [point[1] for point in plot_list_rail_left]

        # rail right x points
        x_coords_right = [point[0] for point in plot_list_rail_right]
        y_coords_right = [point[1] for point in plot_list_rail_right]

        # difference (rail left - rail right)
        x_coords_differ = [diff[0] for diff in plot_list_difference]
        y_coords_differ = [diff[1] for diff in plot_list_difference]

        # min rectangle of rails
        x_coords_min_x = [point[0] for point in plot_list_min_x]
        y_coords_min_x = [point[1] for point in plot_list_min_x]
        x_coords_max_x = [point[0] for point in plot_list_max_x]
        y_coords_max_x = [point[1] for point in plot_list_max_x]

        # auto crops
        # left, top, right, current_frame
        x_coords_auto_crop_left = [point[0] for point in plot_list_auto_crop]
        x_coords_auto_crop_top = [point[1] for point in plot_list_auto_crop]
        x_coords_auto_crop_right = [point[2] for point in plot_list_auto_crop]
        y_coords_auto_crops = [point[3] for point in plot_list_auto_crop]

        # Erstelle das Scatter-Plot mit zwei Linien
        #fig = go.Figure()
        fig = make_subplots(rows=6, cols=1)

        # x points of rails
        fig.add_trace(go.Scatter(x=y_coords_left, y=x_coords_left, mode='lines', name='rail left', line=dict(color='blue')), row=1, col=1) # rail left blue
        fig.add_trace(go.Scatter(x=y_coords_right, y=x_coords_right, mode='lines', name='rail right', line=dict(color='red')), row=1, col=1) # rail right red

        # difference if rails (right - left)
        fig.add_trace(go.Scatter(x=y_coords_differ, y=x_coords_differ, mode='lines', name='rail width', line=dict(color='green')), row=2, col=1) # difference green

        # smallest rectangle of predicted rails
        fig.add_trace(go.Scatter(x=y_coords_min_x, y=x_coords_min_x, mode='lines', name='min x (smallest rec)', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=y_coords_max_x, y=x_coords_max_x, mode='lines', name='max x (smallest rec)', line=dict(color='red')), row=3, col=1)

        # autocrops (left - blue; top - green; right - red)
        fig.add_trace(go.Scatter(x=y_coords_auto_crops, y=x_coords_auto_crop_left, mode='lines', name='auto crop left', line=dict(color='blue')), row=4, col=1)  # left
        fig.add_trace(go.Scatter(x=y_coords_auto_crops, y=x_coords_auto_crop_top, mode='lines', name='auto crop top', line=dict(color='green')), row=4, col=1)   # top
        fig.add_trace(go.Scatter(x=y_coords_auto_crops, y=x_coords_auto_crop_right, mode='lines', name='auto crop right', line=dict(color='red')), row=4, col=1) # right

        # overlay autocrop and smallest rectangle of left border for better visibility
        fig.add_trace(go.Scatter(x=y_coords_min_x, y=x_coords_min_x, mode='lines', name='min x (smallest rec)', line=dict(color='black')), row=5, col=1)
        fig.add_trace(go.Scatter(x=y_coords_auto_crops, y=x_coords_auto_crop_left, mode='lines', name='auto crop left', line=dict(color='green')), row=5, col=1)  # left

        # overlay autocrop and smallest rectangle of right border for better visibility
        fig.add_trace(go.Scatter(x=y_coords_max_x, y=x_coords_max_x, mode='lines', name='max x (smallest rec)', line=dict(color='black')), row=6, col=1)
        fig.add_trace(go.Scatter(x=y_coords_auto_crops, y=x_coords_auto_crop_right, mode='lines', name='auto crop right', line=dict(color='green')), row=6, col=1) # right

        # Layout anpassen (optional)
        #getting backbone name for plot
        backbone_path=os.path.join(base_path, "weights", args.model, "config.yaml") # Load the YAML file
        with open(backbone_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        backbone_name = config.get("backbone") # Access the value associated with the "backbone" key

        fig.update_layout(title=backbone_name, xaxis_title='frame', yaxis_title='predicted X-Coordinate')

        # Plot anzeigen
        fig.show()



    else:
        raise NotImplementedError

    logger.info(f"\nInference complete. Output saved to {output_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

# %%
