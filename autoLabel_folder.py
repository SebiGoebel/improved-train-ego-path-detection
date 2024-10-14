import argparse
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image

from src.utils.common import simple_logger
from src.utils.interface import Detector
from src.utils.filterCoordinates import filterCoords


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

    input_files = os.listdir(args.input)

    image_counter = 1
    for input_file in input_files:
        input_path = os.path.join(args.input, input_file)
        extension = os.path.splitext(input_file)[1]
        outname = f"{os.path.splitext(os.path.basename(input_file))[0]}_out{extension}"

        json_extension = ".json"
        outname_json = f"{os.path.splitext(os.path.basename(input_file))[0]}_out{json_extension}"

        if args.output is not None:
            os.makedirs(args.output, exist_ok=True)
            output_path = os.path.join(args.output, outname)
            print(output_path)
            output_path_json = os.path.join(args.output, outname_json)
        else:
            #output_path = os.path.join(os.path.dirname(args.input), outname)
            #print(output_path)
            #output_path_json = os.path.join(os.path.dirname(args.input), outname_json)
            base_path, last_folder = os.path.split(args.input.rstrip(os.sep))
            new_folder_name = last_folder + "_jsons"
            output_path_json = os.path.join(base_path, new_folder_name) + os.sep
            
            print("json folder path: ", output_path_json)
            os.makedirs(output_path_json, exist_ok=True) # Sicherstellen, dass der neue Ordner existiert

            base_path, last_folder = os.path.split(args.input.rstrip(os.sep))
            new_folder_name = last_folder + "_out"
            output_path_img = os.path.join(base_path, new_folder_name) + os.sep

            print("img_out folder path: ", output_path_img)
            os.makedirs(output_path_img, exist_ok=True) # Sicherstellen, dass der neue Ordner existiert

            output_path = os.path.join(output_path_img, outname)

            print("img_out file path: ", output_path)

        print("------------")
        print(f"image {image_counter} von 67")
        image_counter += 1
        logger.info(f"Detecting ego-path in {input_file}...")

        if extension in [".jpg", ".jpeg", ".png"]:
            frame = Image.open(input_path)
            for _ in range(50 if crop_coords == "auto" else 1):
                crop = detector.get_crop_coords() if args.show_crop else None
                res = detector.detect(frame)
            filterCoords(frame, res, crop_coords=crop, input_file=input_file, output_folder=output_path_json).save(output_path)

        else:
            logger.warning(f"Ignoring {input_file}. Unsupported file format.")

        logger.info(f"Inference complete. Output saved to {output_path}")






    #extension = os.path.splitext(args.input)[1]
    #outname = f"{os.path.splitext(os.path.basename(args.input))[0]}_out{extension}"
    #if args.output is not None:
    #    os.makedirs(args.output, exist_ok=True)
    #    output_path = os.path.join(args.output, outname)
    #else:
    #    output_path = os.path.join(os.path.dirname(args.input), outname)

    #logger.info(f"\nDetecting ego-path in {args.input}...")

    #if extension in [".jpg", ".jpeg", ".png"]:
    #    frame = Image.open(args.input)
    #    for _ in range(50 if crop_coords == "auto" else 1):
    #        crop = detector.get_crop_coords() if args.show_crop else None
    #        res = detector.detect(frame)
    #    draw_egopath(frame, res, crop_coords=crop).save(output_path)

    #elif extension in [".mp4", ".avi"]:
    #    print("wrong detect file!!!!")
    #    print("use detect.py or dectSwitch.py")
    #else:
    #    raise NotImplementedError

    #logger.info(f"\nInference complete. Output saved to {output_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
