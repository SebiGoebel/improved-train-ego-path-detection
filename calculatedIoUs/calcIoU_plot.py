# %%
import numpy as np
import matplotlib.pyplot as plt

lösche_ersten_warmup_indices = 0

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

# datei einlesen
with open("calculateIoU_singleFrame_average_ious_kind-donkey-84_flippedDataset_0_frames_gelöscht.txt", "r") as file:
    ious = [float(line.strip()) for line in file.readlines()]

# IoU
test_iou = np.mean(ious).item()
print(f"Test IoU: {test_iou:.5f}")

ious = np.array(ious) # convert to np array

print("plotting IoUs of the Sequences ...")
# IOU-Liste in 4 Teile aufteilen
iou_parts, means_per_seq = split_into_parts(ious, 76-lösche_ersten_warmup_indices) # through one sequence (76-9 = 67)
# Für jeden Teil einen separaten Plot erstellen
for i, part in enumerate(iou_parts, start=1):
    plot_iou(part, i, means_per_seq[i-1])