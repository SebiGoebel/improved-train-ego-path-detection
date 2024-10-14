"""
Preruns:
bevor man dieses skript ausführen kann müssen 2 txt files exisitieren:
- calculateIoU_singleFrame_average_ious.txt
- calculateIoU_temporal_average_ious.txt

diese bekommt man durch das aufrufen von den 2 python skripte:
- python calculateIoU_singleFrame_temporalDataset.py kind-donkey-84 --device cuda:0
- python calculateIoU_sequence.py stellar-shape-288 --device cuda:0

Das calculateIoU_comparison_singleFrame_2_temporal.py file soll die beiden listen von den average ious aus den .txt files auslesen und ploten.
Legende:
- single-frame-based average ious per frame --> blau
- temporal model average ious per frame     --> rot

ausgeführt wird das skript mit:

python calculateIoU_comparison_singleFrame_2_temporal.py
"""

# Simulate command-line arguments
import sys
sys.argv = ['ipykernel_launcher.py']
# python calculateIoU_comparison_singleFrame_2_temporal.py

import numpy as np
import matplotlib.pyplot as plt

# Funktion zum Einlesen einer Liste aus einer Textdatei
def read_list_from_file(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file.readlines()]


def split_into_parts(ious, part_size=67):
    #return [ious[i:i + part_size] for i in range(0, len(ious), part_size)]
    parts = [ious[i:i + part_size] for i in range(0, len(ious), part_size)]
    mean_values = [np.mean(part, axis=0) for part in parts]  # axis=0, um den Mittelwert über die Zeilen zu berechnen
    return parts, mean_values

def plot_ious(single_frame_based, sequence_based_1, sequence_based_2, sequence_based_3, sequence_num, means_per_seq_single_frame_based, means_per_seq_sequence_based_1, means_per_seq_sequence_based_2, means_per_seq_sequence_based_3):
    if len(single_frame_based) != len(sequence_based_1):
        raise ValueError("die beiden iou listen haben nicht dies selbe Länge!")

    x_values = range(len(single_frame_based)) # Indizes für die x-Achse

    # plotte die Graphik
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, single_frame_based, marker='o', color='blue', label=f'single-frame-based ({means_per_seq_single_frame_based})')
    plt.plot(x_values, sequence_based_1, marker='o', color='red', label=f'RegNetCNN_FC_LSTM ({means_per_seq_sequence_based_1})')
    plt.plot(x_values, sequence_based_2, marker='o', color='green', label=f'RegNetCNN_LSTM ({means_per_seq_sequence_based_2})')
    plt.plot(x_values, sequence_based_3, marker='o', color='orange', label=f'RegNetCNN_FC_FCOUT ({means_per_seq_sequence_based_3})')
    plt.title(f'Comparison between models, test sequence: {sequence_num+1}')
    plt.xlabel('Frame')
    plt.ylabel('IoU')
    plt.legend() # Legende hinzufügen
    plt.grid(True) # Gitterlinien hinzufügen
    plt.show()

# Einlesen der Listen aus den Textdateien
single_frame_based = read_list_from_file('calculateIoU_singleFrame_average_ious.txt')  # Erste Liste                    - single-frame-based
sequence_based_1 = read_list_from_file('calculateIoU_temporal_average_ious_vocal-wildflower-311.txt')  # Zweite Liste   - RegressionNetCNN_FC_LSTM
sequence_based_2 = read_list_from_file('calculateIoU_temporal_average_ious_vital-glade-322.txt')  # Dritte Liste        - RegressionNetCNN_LSTM
sequence_based_3 = read_list_from_file('calculateIoU_temporal_average_ious_vivid-feather-328.txt')  # Vierte Liste      - RegressionNetCNN_FC_FCOUT

# Aufteilen der Listen in die Sequencen
single_frame_based_parts, means_per_seq_single_frame_based = split_into_parts(single_frame_based, 76-10+1)
sequence_based_1_parts, means_per_seq_1_sequence_based = split_into_parts(sequence_based_1, 76-10+1)
sequence_based_2_parts, means_per_seq_2_sequence_based = split_into_parts(sequence_based_2, 76-10+1)
sequence_based_3_parts, means_per_seq_3_sequence_based = split_into_parts(sequence_based_3, 76-10+1)

# Für jeden Teil einen separaten Plot erstellen
for sequence_num in range(len(single_frame_based_parts)):
    plot_ious(single_frame_based_parts[sequence_num], sequence_based_1_parts[sequence_num], sequence_based_2_parts[sequence_num], sequence_based_3_parts[sequence_num], sequence_num, means_per_seq_single_frame_based[sequence_num], means_per_seq_1_sequence_based[sequence_num], means_per_seq_2_sequence_based[sequence_num], means_per_seq_3_sequence_based[sequence_num])