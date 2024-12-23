import timeit

import numpy as np
import torch
from PIL import Image
from torch.utils.benchmark import Timer

from src.utils.common import set_seeds
from src.utils.interface import Detector
from src.utils.postprocessing import rails_to_mask

# visualizing iou plots
import matplotlib.pyplot as plt

plotIoUs = True # um IoUs zu plotten

def compute_iou(input, target):
    """Computes the Intersection over Union (IoU) between two binary masks.

    Args:
        input (numpy.ndarray or PIL.Image.Image): Input mask. Can be a numpy array (True/False, 0/1, 0/255) or a PIL image.
        target (numpy.ndarray or PIL.Image.Image): Ground truth mask. Can be a numpy array (True/False, 0/1, 0/255) or a PIL image.

    Returns:
        float: The IoU score.
    """
    input = np.array(input) if isinstance(input, Image.Image) else input
    target = np.array(target) if isinstance(target, Image.Image) else target
    input = input.astype(bool)
    target = target.astype(bool)
    if np.sum(target) == 0:  # if target is empty, we compute iou on negated masks
        input = np.logical_not(input)
        target = np.logical_not(target)
    intersection = np.logical_and(input, target)
    union = np.logical_or(input, target)
    return (np.sum(intersection) / np.sum(union)).item()

# Funktion zum Aufteilen der IOU-Liste in Blöcke von festgelegter Länge
def split_into_parts(ious, part_size=67):
    #return [ious[i:i + part_size] for i in range(0, len(ious), part_size)]
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

class IoUEvaluator:
    def __init__(self, dataset, model_path, crop, runtime, device):
        """Creates an IoU evaluator for the train ego-path detection model.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to evaluate on.
            model_path (str): Path to the trained model directory (containing config.yaml and best.pt).
            runtime (str): Runtime to use for model inference ("pytorch" or "tensorrt").
            crop (str or None): Coordinates to use for cropping the input image before inference:
                - If str, should be "auto" to use automatic cropping.
                - If None, no cropping is performed.
            device (str): Device to use for model inference ("cpu", "cuda", "cuda:x" or "mps").
        """
        self.dataset = dataset
        self.runtime = runtime

        # crop can only be auto for evaluating with autocropper or None for evaluating different evbal crops, which are done in the dataset class
        if crop == "auto" or crop == None:
            self.crop = crop
        else:
            raise ValueError

        if runtime == "pytorch":
            self.detector = Detector(model_path, crop, runtime, device)
        elif runtime == "tensorrt":
            self.detector = Detector(model_path, crop, runtime, device)
        else:
            raise ValueError
    
    def evaluate(self):
        set_seeds(self.detector.config["seed"])
        ious = []
        # each test epoch is unique due to data augmentation, so we average multiple runs to get a stable result
        for _ in range(1 if self.crop == "auto" else self.detector.config["test_iterations"]): # 10 iterations
            ious_each_iteration = []
            for i in range(len(self.dataset)):
                img, target = self.dataset[i]
                for i in range(450 if self.crop == "auto" else 1): # when autocroppper is selected for evaluation then do 50 iterations  to get a good crop
                    #print(i)
                    pred = self.detector.detect(img)
                if self.detector.config["method"] in ["classification", "regression"]:
                    pred = rails_to_mask(pred, img.size)
                ious_each_iteration.append(compute_iou(pred, target))
            ious.append(ious_each_iteration)

        if plotIoUs:
            # bilde den AVG der 10 iterationen
            ious = np.array(ious)
            avg_ious_each_frame = []
            for col in range(len(ious[0])):  # len(ious[0]) gibt die Anzahl der Spalten (268) zurück
                mean_value = np.mean(ious[:, col])  # ious[:, col] greift auf alle Zeilen der aktuellen Spalte zu
                avg_ious_each_frame.append(mean_value)
            
            print("writing average IoUs to txt file ...")
            with open('calculateIoU_singleFrame_average_ious.txt', 'w') as file:
                for item in avg_ious_each_frame:
                    file.write(f"{item}\n")  # Jeden Wert in einer neuen Zeile schreiben

            print("plotting IoUs of the Sequences ...")
            # IOU-Liste in 4 Teile aufteilen
            avg_iou_parts, means_per_seq = split_into_parts(avg_ious_each_frame, 76-10+1) # through one sequence (76-10+1 = 67)

            # Für jeden Teil einen separaten Plot erstellen
            for i, part in enumerate(avg_iou_parts, start=1):
                plot_iou(part, i, means_per_seq[i-1])
        
        return np.mean(ious).item()
    
class LatencyEvaluator:
    def __init__(self, model_path, runtime, device):
        """Creates a latency evaluator for the train ego-path detection model.

        Args:
            model_path (str): Path to the trained model directory (containing config.yaml and best.pt).
            runtime (str): Runtime environment to use for model inference ("pytorch" or "tensorrt").
            device (str): Device to use for model inference ("cpu", "cuda", "cuda:x" or "mps").
        """        
        self.runtime = runtime
        if runtime == "pytorch":
            self.detector = Detector(model_path, None, runtime, device)
        elif runtime == "tensorrt":
            self.detector = Detector(model_path, None, runtime, device)
        else:
            raise ValueError
        self.device = torch.device(device)

    def evaluate_pytorch(self, runs):
        dummy_input = torch.rand(
            (1, *self.detector.config["input_shape"]), device=self.device
        )
        for _ in range(runs // 10):  # warmup
            self.detector.model(dummy_input)
        timer = Timer(
            stmt="self.detector.model(dummy_input)",
            globals={"torch": torch, "self": self, "dummy_input": dummy_input},
            num_threads=torch.get_num_threads(),
        )
        result = timer.timeit(runs)
        return result.mean  # in seconds

    def evaluate_tensorrt(self, runs):
        dummy_input = np.random.rand(*self.detector.config["input_shape"]).astype(
            np.float32
        )
        self.detector.cuda.memcpy_htod(self.detector.bindings[0], dummy_input)  # to GPU
        for _ in range(runs // 10):  # warmup
            self.detector.exectx.execute_v2(self.detector.bindings)
        timer = timeit.Timer(
            stmt="self.detector.exectx.execute_v2(self.detector.bindings)",
            globals={"self": self},
        )
        return timer.timeit(runs) / runs  # in seconds

    def evaluate(self, runs=1000):
        if self.runtime == "pytorch":
            return self.evaluate_pytorch(runs)
        elif self.runtime == "tensorrt":
            return self.evaluate_tensorrt(runs)
