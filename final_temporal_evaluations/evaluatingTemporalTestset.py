"""
Preruns:
- python detect_IoU.py kind-donkey-84 data/temporalDataset_video.mp4 --show-crop --device cuda:0
- python detect_temporal_IoU.py <temporal Model> data/temporalDataset_video.mp4 --show-crop --device cuda:0

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

lösche_ersten_warmup_indices = 0

# Funktion zum Einlesen einer Liste aus einer Textdatei
def read_list_from_file(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file.readlines()]


def split_into_parts(ious, part_size=76):
    parts = [ious[i:i + part_size] for i in range(0, len(ious), part_size)]
    mean_values = [np.mean(part, axis=0) for part in parts]  # axis=0, um den Mittelwert über die Zeilen zu berechnen

    # Werte in Prozentsatz umwandeln und auf zwei Dezimalstellen runden
    percentage_values = [round(mean_value * 100, 2) for mean_value in mean_values]

    return parts, percentage_values

def plot_ious(single_frame_based,
              sequence_based_1,
              sequence_based_2,
              sequence_based_3,
              sequence_based_4,
              sequence_based_5,
              sequence_based_6,
              sequence_based_7,
              sequence_based_8,
              sequence_based_9,
              sequence_based_10,
              #sequence_based_11,
              sequence_num,
              means_per_seq_single_frame_based,
              means_per_seq_sequence_based_1,
              means_per_seq_sequence_based_2,
              means_per_seq_sequence_based_3,
              means_per_seq_sequence_based_4,
              means_per_seq_sequence_based_5,
              means_per_seq_sequence_based_6,
              means_per_seq_sequence_based_7,
              means_per_seq_sequence_based_8,
              means_per_seq_sequence_based_9,
              means_per_seq_sequence_based_10,
              #means_per_seq_sequence_based_11,
              ):
    if len(single_frame_based) != len(sequence_based_1):
        raise ValueError("die beiden iou listen haben nicht dies selbe Länge!")

    x_values = range(len(single_frame_based)) # Indizes für die x-Achse

    # plotte die Graphik
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, single_frame_based, marker='o', color='blue', label=f'single-frame-based ({means_per_seq_single_frame_based})')
    plt.plot(x_values, sequence_based_1, marker='o', color='red', label=f'CNN_FC_LSTM ({means_per_seq_sequence_based_1})')
    plt.plot(x_values, sequence_based_2, marker='o', color='green', label=f'CNN_LSTM_V1 ({means_per_seq_sequence_based_2})')
    plt.plot(x_values, sequence_based_3, marker='o', color='orange', label=f'CNN_FC_FCOUT_V1 ({means_per_seq_sequence_based_3})')
    plt.plot(x_values, sequence_based_4, marker='o', color='purple', label=f'CNN_LSTM_V2 ({means_per_seq_sequence_based_4})')
    plt.plot(x_values, sequence_based_5, marker='o', color='brown', label=f'CNN_LSTM_HEAD ({means_per_seq_sequence_based_5})')
    plt.plot(x_values, sequence_based_6, marker='o', color='pink', label=f'CNN_FLAT_FC ({means_per_seq_sequence_based_6})')
    plt.plot(x_values, sequence_based_7, marker='o', color='yellow', label=f'CNN_FC_FCOUT_V2 ({means_per_seq_sequence_based_7})')
    plt.plot(x_values, sequence_based_8, marker='o', color='cyan', label=f'CNN_LSTM_SKIP_CAT ({means_per_seq_sequence_based_8})')
    plt.plot(x_values, sequence_based_9, marker='o', color='darkgray', label=f'CNN_LSTM_SKIP_MUL_FEATURE ({means_per_seq_sequence_based_9})')
    plt.plot(x_values, sequence_based_10, marker='o', color='lightgreen', label=f'CNN_LSTM_SKIP_MUL_TIME ({means_per_seq_sequence_based_10})')
    #plt.plot(x_values, sequence_based_11, marker='o', color='teal', label=f'CNN_LSTM_SKIP_MUL_MobileNet ({means_per_seq_sequence_based_11})')
    plt.title(f'Comparison between models, test sequence: {sequence_num+1}')
    plt.xlabel('Frame')
    plt.ylabel('IoU')
    plt.legend() # Legende hinzufügen
    plt.grid(True) # Gitterlinien hinzufügen
    plt.show()

# Einlesen der Listen aus den Textdateien
single_frame_based = read_list_from_file('single-frame-based.txt') # single-frame-based
sequence_based_1 = read_list_from_file('stellar-plant-360.txt')    # CNN_FC_LSTM
sequence_based_2 = read_list_from_file('neat-water-359.txt')       # CNN_LSTM_V1
sequence_based_3 = read_list_from_file('morning-dawn-358.txt')     # CNN_FC_FCOUT_V1
sequence_based_4 = read_list_from_file('dandy-totem-361.txt')      # CNN_LSTM_V2
sequence_based_5 = read_list_from_file('twilight-monkey-362.txt')  # CNN_LSTM_HEAD
sequence_based_6 = read_list_from_file('kind-waterfall-363.txt')   # CNN_FLAT_FC
sequence_based_7 = read_list_from_file('autumn-valley-364.txt')    # CNN_FC_FCOUT_V2
sequence_based_8 = read_list_from_file('silvery-field-386.txt')    # CNN_LSTM_SKIP_CAT
sequence_based_9 = read_list_from_file('trim-pyramid-387.txt')     # CNN_LSTM_SKIP_MUL_FEATURE
sequence_based_10 = read_list_from_file('fanciful-dream-388.txt')  # CNN_LSTM_SKIP_MUL_TIME
#sequence_based_11 = read_list_from_file('dandy-totem-361.txt')    # CNN_LSTM_SKIP_MUL_MobileNet

# Aufteilen der Listen in die Sequencen
single_frame_based_parts, means_per_seq_single_frame_based = split_into_parts(single_frame_based, 76-lösche_ersten_warmup_indices)
sequence_based_1_parts, means_per_seq_1_sequence_based = split_into_parts(sequence_based_1, 76-lösche_ersten_warmup_indices)
sequence_based_2_parts, means_per_seq_2_sequence_based = split_into_parts(sequence_based_2, 76-lösche_ersten_warmup_indices)
sequence_based_3_parts, means_per_seq_3_sequence_based = split_into_parts(sequence_based_3, 76-lösche_ersten_warmup_indices)
sequence_based_4_parts, means_per_seq_4_sequence_based = split_into_parts(sequence_based_4, 76-lösche_ersten_warmup_indices)
sequence_based_5_parts, means_per_seq_5_sequence_based = split_into_parts(sequence_based_5, 76-lösche_ersten_warmup_indices)
sequence_based_6_parts, means_per_seq_6_sequence_based = split_into_parts(sequence_based_6, 76-lösche_ersten_warmup_indices)
sequence_based_7_parts, means_per_seq_7_sequence_based = split_into_parts(sequence_based_7, 76-lösche_ersten_warmup_indices)
sequence_based_8_parts, means_per_seq_8_sequence_based = split_into_parts(sequence_based_8, 76-lösche_ersten_warmup_indices)
sequence_based_9_parts, means_per_seq_9_sequence_based = split_into_parts(sequence_based_9, 76-lösche_ersten_warmup_indices)
sequence_based_10_parts, means_per_seq_10_sequence_based = split_into_parts(sequence_based_10, 76-lösche_ersten_warmup_indices)
#sequence_based_11_parts, means_per_seq_11_sequence_based = split_into_parts(sequence_based_11, 76-lösche_ersten_warmup_indices)

# Für jeden Teil einen separaten Plot erstellen
for sequence_num in range(len(single_frame_based_parts)):
    plot_ious(single_frame_based_parts[sequence_num],
              sequence_based_1_parts[sequence_num],
              sequence_based_2_parts[sequence_num],
              sequence_based_3_parts[sequence_num],
              sequence_based_4_parts[sequence_num],
              sequence_based_5_parts[sequence_num],
              sequence_based_6_parts[sequence_num],
              sequence_based_7_parts[sequence_num],
              sequence_based_8_parts[sequence_num],
              sequence_based_9_parts[sequence_num],
              sequence_based_10_parts[sequence_num],
              #sequence_based_11_parts[sequence_num],
              sequence_num,
              means_per_seq_single_frame_based[sequence_num],
              means_per_seq_1_sequence_based[sequence_num],
              means_per_seq_2_sequence_based[sequence_num],
              means_per_seq_3_sequence_based[sequence_num],
              means_per_seq_4_sequence_based[sequence_num],
              means_per_seq_5_sequence_based[sequence_num],
              means_per_seq_6_sequence_based[sequence_num],
              means_per_seq_7_sequence_based[sequence_num],
              means_per_seq_8_sequence_based[sequence_num],
              means_per_seq_9_sequence_based[sequence_num],
              means_per_seq_10_sequence_based[sequence_num],
              #means_per_seq_11_sequence_based[sequence_num],
              )