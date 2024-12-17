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

# gesamte IoU ausgeben
def calculate_IoU_all_scenes(modlename, values):
    """
    Berechnet das arithmetische Mittel einer Liste und gibt es aus.
    
    :param values: Liste von Zahlen (float oder int)
    """
    if not values:  # Überprüfung, ob die Liste leer ist
        print("list is empty!!!")
        return None

    mean = sum(values) / len(values)
    print(f"IoU on whole testset for {modlename}: {mean:.6f}")
    return mean

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

# Einlesen der Listen aus den Textdateien
single_frame_based = read_list_from_file('single-frame-based.txt') # single-frame-based
single_frame_based_og = read_list_from_file('single-frame-based-original-autocrop.txt') # single-frame-based
original = read_list_from_file('TEP-original.txt') # single-frame-based
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
sequence_based_1_og = read_list_from_file('stellar-plant-360_ogAutocrop.txt')    # CNN_FC_LSTM
sequence_based_2_og = read_list_from_file('neat-water-359_ogAutocrop.txt')       # CNN_LSTM_V1
sequence_based_3_og = read_list_from_file('morning-dawn-358_ogAutocrop.txt')     # CNN_FC_FCOUT_V1
sequence_based_4_og = read_list_from_file('dandy-totem-361_ogAutocrop.txt')      # CNN_LSTM_V2
sequence_based_5_og = read_list_from_file('twilight-monkey-362_ogAutocrop.txt')  # CNN_LSTM_HEAD
sequence_based_6_og = read_list_from_file('kind-waterfall-363_ogAutocrop.txt')   # CNN_FLAT_FC
sequence_based_7_og = read_list_from_file('autumn-valley-364_ogAutocrop.txt')    # CNN_FC_FCOUT_V2
sequence_based_8_og = read_list_from_file('silvery-field-386_ogAutocrop.txt')    # CNN_LSTM_SKIP_CAT
sequence_based_9_og = read_list_from_file('trim-pyramid-387_ogAutocrop.txt')     # CNN_LSTM_SKIP_MUL_FEATURE
sequence_based_10_og = read_list_from_file('fanciful-dream-388_ogAutocrop.txt')  # CNN_LSTM_SKIP_MUL_TIME


#printen der gesamten IoUs pro model
calculate_IoU_all_scenes('TEP-original',                         original)
calculate_IoU_all_scenes('single-frame-based',          single_frame_based)
calculate_IoU_all_scenes('single-frame-based_ogA',          single_frame_based_og)
calculate_IoU_all_scenes('CNN_FC_LSTM',                 sequence_based_1)
calculate_IoU_all_scenes('CNN_LSTM_V1',                 sequence_based_2)
calculate_IoU_all_scenes('CNN_FC_FCOUT_V1',             sequence_based_3)
calculate_IoU_all_scenes('CNN_LSTM_V2',                 sequence_based_4)
calculate_IoU_all_scenes('CNN_LSTM_HEAD',               sequence_based_5)
calculate_IoU_all_scenes('CNN_FLAT_FC',                 sequence_based_6)
calculate_IoU_all_scenes('CNN_FC_FCOUT_V2',             sequence_based_7)
calculate_IoU_all_scenes('CNN_LSTM_SKIP_CAT',           sequence_based_8)
calculate_IoU_all_scenes('CNN_LSTM_SKIP_MUL_FEATURE',   sequence_based_9)
calculate_IoU_all_scenes('CNN_LSTM_SKIP_MUL_TIME',      sequence_based_10)
calculate_IoU_all_scenes('CNN_FC_LSTM_ogA',                 sequence_based_1_og)
calculate_IoU_all_scenes('CNN_LSTM_V1_ogA',                 sequence_based_2_og)
calculate_IoU_all_scenes('CNN_FC_FCOUT_V1_ogA',             sequence_based_3_og)
calculate_IoU_all_scenes('CNN_LSTM_V2_ogA',                 sequence_based_4_og)
calculate_IoU_all_scenes('CNN_LSTM_HEAD_ogA',               sequence_based_5_og)
calculate_IoU_all_scenes('CNN_FLAT_FC_ogA',                 sequence_based_6_og)
calculate_IoU_all_scenes('CNN_FC_FCOUT_V2_ogA',             sequence_based_7_og)
calculate_IoU_all_scenes('CNN_LSTM_SKIP_CAT_ogA',           sequence_based_8_og)
calculate_IoU_all_scenes('CNN_LSTM_SKIP_MUL_FEATURE_ogA',   sequence_based_9_og)
calculate_IoU_all_scenes('CNN_LSTM_SKIP_MUL_TIME_ogA',      sequence_based_10_og)

# Aufteilen der Listen in die Sequencen
single_frame_based_parts, means_per_seq_single_frame_based = split_into_parts(single_frame_based, 76-lösche_ersten_warmup_indices)
single_frame_based_parts_og, means_per_seq_single_frame_based_og = split_into_parts(single_frame_based_og, 76-lösche_ersten_warmup_indices)
original_parts, means_per_seq_original                  = split_into_parts(original, 76-lösche_ersten_warmup_indices)
sequence_based_1_parts, means_per_seq_1_sequence_based  = split_into_parts(sequence_based_1, 76-lösche_ersten_warmup_indices)
sequence_based_2_parts, means_per_seq_2_sequence_based  = split_into_parts(sequence_based_2, 76-lösche_ersten_warmup_indices)
sequence_based_3_parts, means_per_seq_3_sequence_based  = split_into_parts(sequence_based_3, 76-lösche_ersten_warmup_indices)
sequence_based_4_parts, means_per_seq_4_sequence_based  = split_into_parts(sequence_based_4, 76-lösche_ersten_warmup_indices)
sequence_based_5_parts, means_per_seq_5_sequence_based  = split_into_parts(sequence_based_5, 76-lösche_ersten_warmup_indices)
sequence_based_6_parts, means_per_seq_6_sequence_based  = split_into_parts(sequence_based_6, 76-lösche_ersten_warmup_indices)
sequence_based_7_parts, means_per_seq_7_sequence_based  = split_into_parts(sequence_based_7, 76-lösche_ersten_warmup_indices)
sequence_based_8_parts, means_per_seq_8_sequence_based  = split_into_parts(sequence_based_8, 76-lösche_ersten_warmup_indices)
sequence_based_9_parts, means_per_seq_9_sequence_based  = split_into_parts(sequence_based_9, 76-lösche_ersten_warmup_indices)
sequence_based_10_parts, means_per_seq_10_sequence_based = split_into_parts(sequence_based_10, 76-lösche_ersten_warmup_indices)
sequence_based_1_parts_og, means_per_seq_1_sequence_based_og    = split_into_parts(sequence_based_1_og, 76-lösche_ersten_warmup_indices)
sequence_based_2_parts_og, means_per_seq_2_sequence_based_og    = split_into_parts(sequence_based_2_og, 76-lösche_ersten_warmup_indices)
sequence_based_3_parts_og, means_per_seq_3_sequence_based_og    = split_into_parts(sequence_based_3_og, 76-lösche_ersten_warmup_indices)
sequence_based_4_parts_og, means_per_seq_4_sequence_based_og    = split_into_parts(sequence_based_4_og, 76-lösche_ersten_warmup_indices)
sequence_based_5_parts_og, means_per_seq_5_sequence_based_og    = split_into_parts(sequence_based_5_og, 76-lösche_ersten_warmup_indices)
sequence_based_6_parts_og, means_per_seq_6_sequence_based_og    = split_into_parts(sequence_based_6_og, 76-lösche_ersten_warmup_indices)
sequence_based_7_parts_og, means_per_seq_7_sequence_based_og    = split_into_parts(sequence_based_7_og, 76-lösche_ersten_warmup_indices)
sequence_based_8_parts_og, means_per_seq_8_sequence_based_og    = split_into_parts(sequence_based_8_og, 76-lösche_ersten_warmup_indices)
sequence_based_9_parts_og, means_per_seq_9_sequence_based_og    = split_into_parts(sequence_based_9_og, 76-lösche_ersten_warmup_indices)
sequence_based_10_parts_og, means_per_seq_10_sequence_based_og  = split_into_parts(sequence_based_10_og, 76-lösche_ersten_warmup_indices)

switch_drive_by = [(34, 64),
                   (33, 75), #(32, 65),
                   (31, 75),
                   (7, 51)]

# =============================================== Sequence 1 ===============================================

# Für jeden Teil einen separaten Plot erstellen
sequence_num = 0
if len(single_frame_based_parts[sequence_num]) != len(sequence_based_1_parts[sequence_num]):
        raise ValueError("die beiden iou listen haben nicht dies selbe Länge!")

x_values = range(len(single_frame_based_parts[sequence_num])) # Indizes für die x-Achse

# plotte die Graphik
plt.figure(figsize=(10, 6))

# Hintergrund einfärben
highlight_start, highlight_end = switch_drive_by[sequence_num]
plt.axvspan(highlight_start, highlight_end, color='green', alpha=0.3, label='switch')
plt.plot(x_values, sequence_based_1_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_2_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_3_parts[sequence_num]         , marker='.', color='brown')
plt.plot(x_values, sequence_based_5_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_6_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_7_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_8_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_9_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_1_parts_og[sequence_num]      , marker='.', color='cyan')
plt.plot(x_values, sequence_based_2_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_3_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_4_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_5_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_6_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_7_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_8_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_9_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_10_parts_og[sequence_num]     , marker='.', color='darkgray')
plt.plot(x_values, original_parts[sequence_num]                 , marker='.', color='blue', label=f'original')
plt.plot(x_values, single_frame_based_parts[sequence_num]       , marker='.', color='orange', label=f'single-frame-based')
plt.plot(x_values, single_frame_based_parts_og[sequence_num]    , marker='.', color='green', label=f'single-frame-based_ogA')
plt.plot(x_values, sequence_based_10_parts[sequence_num]        , marker='.', color='red', label=f'CNN_LSTM_SKIP_MUL_TIME')
plt.plot(x_values, sequence_based_4_parts[sequence_num]         , marker='.', color='purple', label=f'CNN_LSTM_HEAD')
plt.xlabel('Frame')
plt.ylabel('IoU')
plt.legend() # Legende hinzufügen
plt.grid(True) # Gitterlinien hinzufügen
plt.savefig(f"../imagesFilesForMasterthesis/plot_ious_sequence_{sequence_num+1}.svg", format="svg") # speichern als svg
print(f"saved plot_ious_sequence_{sequence_num+1}.svg")
plt.show()

# =============================================== Sequence 2 ===============================================

# Für jeden Teil einen separaten Plot erstellen
sequence_num = 1
if len(single_frame_based_parts[sequence_num]) != len(sequence_based_1_parts[sequence_num]):
        raise ValueError("die beiden iou listen haben nicht dies selbe Länge!")

x_values = range(len(single_frame_based_parts[sequence_num])) # Indizes für die x-Achse

# plotte die Graphik
plt.figure(figsize=(10, 6))

# Hintergrund einfärben
highlight_start, highlight_end = switch_drive_by[sequence_num]
plt.axvspan(highlight_start, highlight_end, color='green', alpha=0.3, label='switch')
plt.plot(x_values, sequence_based_1_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_2_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_3_parts[sequence_num]         , marker='.', color='brown')
plt.plot(x_values, sequence_based_4_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_5_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_6_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_7_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_8_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_9_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_10_parts[sequence_num]        , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_2_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_3_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_4_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_5_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_6_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_7_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_8_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_9_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_10_parts_og[sequence_num]     , marker='.', color='darkgray')
plt.plot(x_values, original_parts[sequence_num]                 , marker='.', color='blue', label=f'original')
plt.plot(x_values, single_frame_based_parts[sequence_num]       , marker='.', color='orange', label=f'single-frame-based')
plt.plot(x_values, single_frame_based_parts_og[sequence_num]    , marker='.', color='green', label=f'single-frame-based_ogA')
plt.plot(x_values, sequence_based_1_parts_og[sequence_num]      , marker='.', color='red', label=f'CNN_FC_LSTM_ogA')
plt.xlabel('Frame')
plt.ylabel('IoU')
plt.legend() # Legende hinzufügen
plt.grid(True) # Gitterlinien hinzufügen
plt.savefig(f"../imagesFilesForMasterthesis/plot_ious_sequence_{sequence_num+1}.svg", format="svg") # speichern als svg
print(f"saved plot_ious_sequence_{sequence_num+1}.svg")
plt.show()

# =============================================== Sequence 3 ===============================================

# Für jeden Teil einen separaten Plot erstellen
sequence_num = 2
if len(single_frame_based_parts[sequence_num]) != len(sequence_based_1_parts[sequence_num]):
        raise ValueError("die beiden iou listen haben nicht dies selbe Länge!")

x_values = range(len(single_frame_based_parts[sequence_num])) # Indizes für die x-Achse

# plotte die Graphik
plt.figure(figsize=(10, 6))

# Hintergrund einfärben
highlight_start, highlight_end = switch_drive_by[sequence_num]
plt.axvspan(highlight_start, highlight_end, color='green', alpha=0.3, label='switch')
plt.plot(x_values, sequence_based_1_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_3_parts[sequence_num]         , marker='.', color='brown')
plt.plot(x_values, sequence_based_4_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_5_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_6_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_7_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_8_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_9_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_10_parts[sequence_num]        , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_1_parts_og[sequence_num]      , marker='.', color='cyan')
plt.plot(x_values, sequence_based_2_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_3_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_4_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_5_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_6_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_7_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_8_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_9_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_10_parts_og[sequence_num]     , marker='.', color='darkgray')
plt.plot(x_values, original_parts[sequence_num]                 , marker='.', color='blue', label=f'original')
plt.plot(x_values, single_frame_based_parts[sequence_num]       , marker='.', color='orange', label=f'single-frame-based')
plt.plot(x_values, single_frame_based_parts_og[sequence_num]    , marker='.', color='green', label=f'single-frame-based_ogA')
plt.plot(x_values, sequence_based_2_parts[sequence_num]         , marker='.', color='red', label=f'CNN_LSTM_V1')
plt.xlabel('Frame')
plt.ylabel('IoU')
plt.legend() # Legende hinzufügen
plt.grid(True) # Gitterlinien hinzufügen
plt.savefig(f"../imagesFilesForMasterthesis/plot_ious_sequence_{sequence_num+1}.svg", format="svg") # speichern als svg
print(f"saved plot_ious_sequence_{sequence_num+1}.svg")
plt.show()

# =============================================== Sequence 4 ===============================================

# Für jeden Teil einen separaten Plot erstellen
sequence_num = 3
if len(single_frame_based_parts[sequence_num]) != len(sequence_based_1_parts[sequence_num]):
        raise ValueError("die beiden iou listen haben nicht dies selbe Länge!")

x_values = range(len(single_frame_based_parts[sequence_num])) # Indizes für die x-Achse

# plotte die Graphik
plt.figure(figsize=(10, 6))

# Hintergrund einfärben
highlight_start, highlight_end = switch_drive_by[sequence_num]
plt.axvspan(highlight_start, highlight_end, color='green', alpha=0.3, label='switch')
plt.plot(x_values, sequence_based_1_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_2_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_3_parts[sequence_num]         , marker='.', color='brown')
plt.plot(x_values, sequence_based_4_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_5_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_6_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_7_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_8_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_9_parts[sequence_num]         , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_10_parts[sequence_num]        , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_2_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_3_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_4_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_5_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_6_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_7_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_8_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_9_parts_og[sequence_num]      , marker='.', color='darkgray')
plt.plot(x_values, sequence_based_10_parts_og[sequence_num]     , marker='.', color='darkgray')
plt.plot(x_values, original_parts[sequence_num]                 , marker='.', color='blue', label=f'original')
plt.plot(x_values, single_frame_based_parts[sequence_num]       , marker='.', color='orange', label=f'single-frame-based')
plt.plot(x_values, single_frame_based_parts_og[sequence_num]    , marker='.', color='green', label=f'single-frame-based_ogA')
plt.plot(x_values, sequence_based_1_parts_og[sequence_num]      , marker='.', color='red', label=f'CNN_FC_LSTM_ogA')
plt.xlabel('Frame')
plt.ylabel('IoU')
plt.legend() # Legende hinzufügen
plt.grid(True) # Gitterlinien hinzufügen
plt.savefig(f"../imagesFilesForMasterthesis/plot_ious_sequence_{sequence_num+1}.svg", format="svg") # speichern als svg
print(f"saved plot_ious_sequence_{sequence_num+1}.svg")
plt.show()
