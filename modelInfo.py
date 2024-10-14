"""
python modelInfo.py hardy-wildflower-52 data/egopath.jpg --device cuda:0
"""

import argparse
import os

import torch.utils
import torch.utils.flop_counter

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image

from src.utils.common import simple_logger
from src.utils.interface import Detector
from src.utils.visualization import draw_egopath

from torchsummary import summary
#import torchinfo
from thop import profile


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

    detector.model.eval()
    #summary(detector.model, (3, 512, 512))
    dummyInputTensor = torch.empty(1, 3, 512, 512, device=args.device)

    # Define the path to save the ONNX file
    #onnx_file_path = 'model.onnx'

    onnx_file_name = f"{os.path.splitext(args.model)[0]}.onnx"
    onnx_file_path = os.path.join(base_path, "weights", args.model, onnx_file_name)
    print("onnx_file_path")
    print(onnx_file_path)

    # Export the model
    torch.onnx.export(
        detector.model,          # The model to be exported
        dummyInputTensor,        # Dummy input to the model
        onnx_file_path,          # Output file path
        export_params=True,      # Store the trained parameter weights inside the model file
        opset_version=11,        # ONNX version to export the model to
        do_constant_folding=True # Whether to execute constant folding for optimization
    )

    print(f"Model has been converted to ONNX and saved at {onnx_file_path}")

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in detector.model.state_dict():
        print(param_tensor, "\t", detector.model.state_dict()[param_tensor].size())


    macs, params = profile(detector.model, inputs=(dummyInputTensor,))
    print("---------------------------------------------------")
    print("MACs: ", macs)
    print("params: ", params)
    print("---------------------------------------------------")
    flops = torch.utils.flop_counter.FlopCounterMode(detector.model, depth=1)
    with flops:
        detector.model(dummyInputTensor).sum()

    #torchinfo funktioniert nicht
    #torchinfo.summary(detector.model.cuda(), (3, 512, 512), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose = 0)

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
        progress_bar = simple_logger(f"{__name__}_progress", "info", terminator="\r")
        while current_frame < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            crop = detector.get_crop_coords() if args.show_crop else None
            res = detector.detect(frame)
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

    else:
        raise NotImplementedError

    logger.info(f"\nInference complete. Output saved to {output_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
