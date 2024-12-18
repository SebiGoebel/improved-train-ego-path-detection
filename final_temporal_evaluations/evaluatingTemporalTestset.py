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

def plot_ious(original,
              single_frame_based,
              single_frame_based_og,
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
              sequence_based_1_og,
              sequence_based_2_og,
              sequence_based_3_og,
              sequence_based_4_og,
              sequence_based_5_og,
              sequence_based_6_og,
              sequence_based_7_og,
              sequence_based_8_og,
              sequence_based_9_og,
              sequence_based_10_og,
              sequence_num,
              means_per_seq_original,
              means_per_seq_single_frame_based,
              means_per_seq_single_frame_based_og,
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
              means_per_seq_sequence_based_1_og,
              means_per_seq_sequence_based_2_og,
              means_per_seq_sequence_based_3_og,
              means_per_seq_sequence_based_4_og,
              means_per_seq_sequence_based_5_og,
              means_per_seq_sequence_based_6_og,
              means_per_seq_sequence_based_7_og,
              means_per_seq_sequence_based_8_og,
              means_per_seq_sequence_based_9_og,
              means_per_seq_sequence_based_10_og,
              highlight_area,
              ):
    if len(single_frame_based) != len(sequence_based_1):
        raise ValueError("die beiden iou listen haben nicht dies selbe Länge!")

    x_values = range(len(single_frame_based)) # Indizes für die x-Achse

    # plotte die Graphik
    plt.figure(figsize=(10, 6))

    # Hintergrund einfärben
    highlight_start, highlight_end = highlight_area
    plt.axvspan(highlight_start, highlight_end, color='green', alpha=0.3, label='switch')

    #plt.plot(x_values, single_frame_based, marker='.', color='blue', label=f'single-frame-based ({means_per_seq_single_frame_based})')
    #plt.plot(x_values, sequence_based_1, marker='.', color='red', label=f'CNN_FC_LSTM ({means_per_seq_sequence_based_1})')
    #plt.plot(x_values, sequence_based_2, marker='.', color='green', label=f'CNN_LSTM_V1 ({means_per_seq_sequence_based_2})')
    #plt.plot(x_values, sequence_based_4, marker='.', color='purple', label=f'CNN_LSTM_V2 ({means_per_seq_sequence_based_4})')
    #plt.plot(x_values, sequence_based_5, marker='.', color='brown', label=f'CNN_LSTM_HEAD ({means_per_seq_sequence_based_5})')
    #plt.plot(x_values, sequence_based_3, marker='.', color='orange', label=f'CNN_FC_FCOUT_V1 ({means_per_seq_sequence_based_3})')
    #plt.plot(x_values, sequence_based_7, marker='.', color='yellow', label=f'CNN_FC_FCOUT_V2 ({means_per_seq_sequence_based_7})')
    #plt.plot(x_values, sequence_based_6, marker='.', color='pink', label=f'CNN_FLAT_FC ({means_per_seq_sequence_based_6})')
    #plt.plot(x_values, sequence_based_8, marker='.', color='cyan', label=f'CNN_LSTM_SKIP_CAT ({means_per_seq_sequence_based_8})')
    #plt.plot(x_values, sequence_based_9, marker='.', color='darkgray', label=f'CNN_LSTM_SKIP_MUL_FEATURE ({means_per_seq_sequence_based_9})')
    #plt.plot(x_values, sequence_based_10, marker='.', color='lightgreen', label=f'CNN_LSTM_SKIP_MUL_TIME ({means_per_seq_sequence_based_10})')
    

    plt.plot(x_values, original, marker='.', color='teal', label=f'original                              ({means_per_seq_original})')
    plt.plot(x_values, single_frame_based, marker='.', color='blue', label=f'single-frame-based          ({means_per_seq_single_frame_based})')
    plt.plot(x_values, single_frame_based_og, marker='.', color='blue', label=f'single-frame-based_ogA      ({means_per_seq_single_frame_based_og})')
    plt.plot(x_values, sequence_based_1, marker='.', color='red',    label=f'CNN_FC_LSTM                 ({means_per_seq_sequence_based_1})')
    plt.plot(x_values, sequence_based_2, marker='.', color='green',  label=f'CNN_LSTM_V1                 ({means_per_seq_sequence_based_2})')
    plt.plot(x_values, sequence_based_4, marker='.', color='purple', label=f'CNN_LSTM_V2                 ({means_per_seq_sequence_based_4})')
    plt.plot(x_values, sequence_based_5, marker='.', color='brown', label=f'CNN_LSTM_HEAD                ({means_per_seq_sequence_based_5})')
    plt.plot(x_values, sequence_based_3, marker='.', color='orange', label=f'CNN_FC_FCOUT_V1             ({means_per_seq_sequence_based_3})')
    plt.plot(x_values, sequence_based_7, marker='.', color='yellow', label=f'CNN_FC_FCOUT_V2             ({means_per_seq_sequence_based_7})')
    plt.plot(x_values, sequence_based_6, marker='.', color='pink', label=f'CNN_FLAT_FC                   ({means_per_seq_sequence_based_6})')
    plt.plot(x_values, sequence_based_8, marker='.', color='cyan', label=f'CNN_LSTM_SKIP_CAT             ({means_per_seq_sequence_based_8})')
    plt.plot(x_values, sequence_based_9, marker='.', color='darkgray', label=f'CNN_LSTM_SKIP_MUL_FEATURE ({means_per_seq_sequence_based_9})')
    plt.plot(x_values, sequence_based_10, marker='.', color='lightgreen', label=f'CNN_LSTM_SKIP_MUL_TIME ({means_per_seq_sequence_based_10})')
    plt.plot(x_values, sequence_based_1_og, marker='.', color='red',    label=f'CNN_FC_LSTM_ogA                 ({means_per_seq_sequence_based_1_og})')
    plt.plot(x_values, sequence_based_2_og, marker='.', color='green',  label=f'CNN_LSTM_V1_ogA                 ({means_per_seq_sequence_based_2_og})')
    plt.plot(x_values, sequence_based_4_og, marker='.', color='purple', label=f'CNN_LSTM_V2_ogA                 ({means_per_seq_sequence_based_4_og})')
    plt.plot(x_values, sequence_based_5_og, marker='.', color='brown', label=f'CNN_LSTM_HEAD_ogA                ({means_per_seq_sequence_based_5_og})')
    plt.plot(x_values, sequence_based_3_og, marker='.', color='orange', label=f'CNN_FC_FCOUT_V1_ogA             ({means_per_seq_sequence_based_3_og})')
    plt.plot(x_values, sequence_based_7_og, marker='.', color='yellow', label=f'CNN_FC_FCOUT_V2_ogA             ({means_per_seq_sequence_based_7_og})')
    plt.plot(x_values, sequence_based_6_og, marker='.', color='pink', label=f'CNN_FLAT_FC_ogA                   ({means_per_seq_sequence_based_6_og})')
    plt.plot(x_values, sequence_based_8_og, marker='.', color='cyan', label=f'CNN_LSTM_SKIP_CAT_ogA             ({means_per_seq_sequence_based_8_og})')
    plt.plot(x_values, sequence_based_9_og, marker='.', color='darkgray', label=f'CNN_LSTM_SKIP_MUL_FEATURE_ogA ({means_per_seq_sequence_based_9_og})')
    plt.plot(x_values, sequence_based_10_og, marker='.', color='lightgreen', label=f'CNN_LSTM_SKIP_MUL_TIME_ogA ({means_per_seq_sequence_based_10_og})')
    plt.title(f'Comparison between models, test sequence: {sequence_num+1}')
    plt.xlabel('Frame')
    plt.ylabel('IoU')
    plt.legend() # Legende hinzufügen
    plt.grid(True) # Gitterlinien hinzufügen
    plt.savefig(f"../imagesFilesForMasterthesis/plot_ious_sequence_{sequence_num + 1}.svg", format="svg") # speichern als svg
    print(f"saved plot_ious_sequence_{sequence_num + 1}.svg")
    plt.show()



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
calculate_IoU_all_scenes('TEP-original',                    original)
calculate_IoU_all_scenes('single-frame-based',              single_frame_based)
calculate_IoU_all_scenes('single-frame-based_ogA',          single_frame_based_og)
calculate_IoU_all_scenes('CNN_FC_LSTM',                     sequence_based_1)
calculate_IoU_all_scenes('CNN_LSTM_V1',                     sequence_based_2)
calculate_IoU_all_scenes('CNN_FC_FCOUT_V1',                 sequence_based_3)
calculate_IoU_all_scenes('CNN_LSTM_V2',                     sequence_based_4)
calculate_IoU_all_scenes('CNN_LSTM_HEAD',                   sequence_based_5)
calculate_IoU_all_scenes('CNN_FLAT_FC',                     sequence_based_6)
calculate_IoU_all_scenes('CNN_FC_FCOUT_V2',                 sequence_based_7)
calculate_IoU_all_scenes('CNN_LSTM_SKIP_CAT',               sequence_based_8)
calculate_IoU_all_scenes('CNN_LSTM_SKIP_MUL_FEATURE',       sequence_based_9)
calculate_IoU_all_scenes('CNN_LSTM_SKIP_MUL_TIME',          sequence_based_10)
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
original_parts, means_per_seq_original = split_into_parts(original, 76-lösche_ersten_warmup_indices)
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
sequence_based_1_parts_og, means_per_seq_1_sequence_based_og = split_into_parts(sequence_based_1_og, 76-lösche_ersten_warmup_indices)
sequence_based_2_parts_og, means_per_seq_2_sequence_based_og = split_into_parts(sequence_based_2_og, 76-lösche_ersten_warmup_indices)
sequence_based_3_parts_og, means_per_seq_3_sequence_based_og = split_into_parts(sequence_based_3_og, 76-lösche_ersten_warmup_indices)
sequence_based_4_parts_og, means_per_seq_4_sequence_based_og = split_into_parts(sequence_based_4_og, 76-lösche_ersten_warmup_indices)
sequence_based_5_parts_og, means_per_seq_5_sequence_based_og = split_into_parts(sequence_based_5_og, 76-lösche_ersten_warmup_indices)
sequence_based_6_parts_og, means_per_seq_6_sequence_based_og = split_into_parts(sequence_based_6_og, 76-lösche_ersten_warmup_indices)
sequence_based_7_parts_og, means_per_seq_7_sequence_based_og = split_into_parts(sequence_based_7_og, 76-lösche_ersten_warmup_indices)
sequence_based_8_parts_og, means_per_seq_8_sequence_based_og = split_into_parts(sequence_based_8_og, 76-lösche_ersten_warmup_indices)
sequence_based_9_parts_og, means_per_seq_9_sequence_based_og = split_into_parts(sequence_based_9_og, 76-lösche_ersten_warmup_indices)
sequence_based_10_parts_og, means_per_seq_10_sequence_based_og = split_into_parts(sequence_based_10_og, 76-lösche_ersten_warmup_indices)

switch_drive_by = [(34, 64),
                   (33, 75), #(32, 65),
                   (31, 75),
                   (7, 51)]

list_for_switch_values_original = []
list_for_switch_values_single = []
list_for_switch_values_single_og = []
list_for_switch_values_1 = []
list_for_switch_values_2 = []
list_for_switch_values_3 = []
list_for_switch_values_4 = []
list_for_switch_values_5 = []
list_for_switch_values_6 = []
list_for_switch_values_7 = []
list_for_switch_values_8 = []
list_for_switch_values_9 = []
list_for_switch_values_10 = []
list_for_switch_values_1_og = []
list_for_switch_values_2_og = []
list_for_switch_values_3_og = []
list_for_switch_values_4_og = []
list_for_switch_values_5_og = []
list_for_switch_values_6_og = []
list_for_switch_values_7_og = []
list_for_switch_values_8_og = []
list_for_switch_values_9_og = []
list_for_switch_values_10_og = []
#list_for_switch_values_11 = []

# Für jeden Teil einen separaten Plot erstellen
for sequence_num in range(len(single_frame_based_parts)):
    plot_ious(original_parts[sequence_num],
              single_frame_based_parts[sequence_num],
              single_frame_based_parts_og[sequence_num],
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
              sequence_based_1_parts_og[sequence_num],
              sequence_based_2_parts_og[sequence_num],
              sequence_based_3_parts_og[sequence_num],
              sequence_based_4_parts_og[sequence_num],
              sequence_based_5_parts_og[sequence_num],
              sequence_based_6_parts_og[sequence_num],
              sequence_based_7_parts_og[sequence_num],
              sequence_based_8_parts_og[sequence_num],
              sequence_based_9_parts_og[sequence_num],
              sequence_based_10_parts_og[sequence_num],
              sequence_num,
              means_per_seq_original[sequence_num],
              means_per_seq_single_frame_based[sequence_num],
              means_per_seq_single_frame_based_og[sequence_num],
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
              means_per_seq_1_sequence_based_og[sequence_num],
              means_per_seq_2_sequence_based_og[sequence_num],
              means_per_seq_3_sequence_based_og[sequence_num],
              means_per_seq_4_sequence_based_og[sequence_num],
              means_per_seq_5_sequence_based_og[sequence_num],
              means_per_seq_6_sequence_based_og[sequence_num],
              means_per_seq_7_sequence_based_og[sequence_num],
              means_per_seq_8_sequence_based_og[sequence_num],
              means_per_seq_9_sequence_based_og[sequence_num],
              means_per_seq_10_sequence_based_og[sequence_num],
              #means_per_seq_11_sequence_based[sequence_num],
              switch_drive_by[sequence_num],
              )
    switch_start, switch_end = switch_drive_by[sequence_num]
    mean_value_original = np.mean(original_parts[sequence_num][switch_start:switch_end])
    mean_value_single_frame = np.mean(single_frame_based_parts[sequence_num][switch_start:switch_end])
    mean_value_single_frame_og = np.mean(single_frame_based_parts_og[sequence_num][switch_start:switch_end])
    mean_value_sequence_1 = np.mean(sequence_based_1_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_2 = np.mean(sequence_based_2_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_3 = np.mean(sequence_based_3_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_4 = np.mean(sequence_based_4_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_5 = np.mean(sequence_based_5_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_6 = np.mean(sequence_based_6_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_7 = np.mean(sequence_based_7_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_8 = np.mean(sequence_based_8_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_9 = np.mean(sequence_based_9_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_10 = np.mean(sequence_based_10_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_1_og = np.mean(sequence_based_1_parts_og[sequence_num][switch_start:switch_end])
    mean_value_sequence_2_og = np.mean(sequence_based_2_parts_og[sequence_num][switch_start:switch_end])
    mean_value_sequence_3_og = np.mean(sequence_based_3_parts_og[sequence_num][switch_start:switch_end])
    mean_value_sequence_4_og = np.mean(sequence_based_4_parts_og[sequence_num][switch_start:switch_end])
    mean_value_sequence_5_og = np.mean(sequence_based_5_parts_og[sequence_num][switch_start:switch_end])
    mean_value_sequence_6_og = np.mean(sequence_based_6_parts_og[sequence_num][switch_start:switch_end])
    mean_value_sequence_7_og = np.mean(sequence_based_7_parts_og[sequence_num][switch_start:switch_end])
    mean_value_sequence_8_og = np.mean(sequence_based_8_parts_og[sequence_num][switch_start:switch_end])
    mean_value_sequence_9_og = np.mean(sequence_based_9_parts_og[sequence_num][switch_start:switch_end])
    mean_value_sequence_10_og = np.mean(sequence_based_10_parts_og[sequence_num][switch_start:switch_end])
    list_for_switch_values_original.append(mean_value_original)
    list_for_switch_values_single.append(mean_value_single_frame)
    list_for_switch_values_single_og.append(mean_value_single_frame_og)
    list_for_switch_values_1.append(mean_value_sequence_1)
    list_for_switch_values_2.append(mean_value_sequence_2)
    list_for_switch_values_3.append(mean_value_sequence_3)
    list_for_switch_values_4.append(mean_value_sequence_4)
    list_for_switch_values_5.append(mean_value_sequence_5)
    list_for_switch_values_6.append(mean_value_sequence_6)
    list_for_switch_values_7.append(mean_value_sequence_7)
    list_for_switch_values_8.append(mean_value_sequence_8)
    list_for_switch_values_9.append(mean_value_sequence_9)
    list_for_switch_values_10.append(mean_value_sequence_10)
    list_for_switch_values_1_og.append(mean_value_sequence_1_og)
    list_for_switch_values_2_og.append(mean_value_sequence_2_og)
    list_for_switch_values_3_og.append(mean_value_sequence_3_og)
    list_for_switch_values_4_og.append(mean_value_sequence_4_og)
    list_for_switch_values_5_og.append(mean_value_sequence_5_og)
    list_for_switch_values_6_og.append(mean_value_sequence_6_og)
    list_for_switch_values_7_og.append(mean_value_sequence_7_og)
    list_for_switch_values_8_og.append(mean_value_sequence_8_og)
    list_for_switch_values_9_og.append(mean_value_sequence_9_og)
    list_for_switch_values_10_og.append(mean_value_sequence_10_og)
    
    #list_for_switch_values_single.append(single_frame_based_parts[sequence_num][switch_start:switch_end])
    #list_for_switch_values_1.append(sequence_based_1_parts[sequence_num][switch_start:switch_end])
    #list_for_switch_values_2.append(sequence_based_2_parts[sequence_num][switch_start:switch_end])
    #list_for_switch_values_3.append(sequence_based_3_parts[sequence_num][switch_start:switch_end])
    #list_for_switch_values_4.append(sequence_based_4_parts[sequence_num][switch_start:switch_end])
    #list_for_switch_values_5.append(sequence_based_5_parts[sequence_num][switch_start:switch_end])
    #list_for_switch_values_6.append(sequence_based_6_parts[sequence_num][switch_start:switch_end])
    #list_for_switch_values_7.append(sequence_based_7_parts[sequence_num][switch_start:switch_end])
    #list_for_switch_values_8.append(sequence_based_8_parts[sequence_num][switch_start:switch_end])
    #list_for_switch_values_9.append(sequence_based_9_parts[sequence_num][switch_start:switch_end])
    #list_for_switch_values_10.append(sequence_based_10_parts[sequence_num][switch_start:switch_end])
    
    
    
    
    print("test sequence: ", sequence_num+1)
    print("original:                  ", round(mean_value_original * 100, 2))
    print("single-frame-based:        ", round(mean_value_single_frame * 100, 2))
    print("single-frame-based_ogA:        ", round(mean_value_single_frame_og * 100, 2))
    print("CNN_FC_LSTM:               ", round(mean_value_sequence_1 * 100, 2))
    print("CNN_LSTM_V1:               ", round(mean_value_sequence_2 * 100, 2))
    print("CNN_LSTM_V2:               ", round(mean_value_sequence_4 * 100, 2))
    print("CNN_LSTM_HEAD:             ", round(mean_value_sequence_5 * 100, 2))
    print("CNN_FC_FCOUT_V1:           ", round(mean_value_sequence_3 * 100, 2))
    print("CNN_FC_FCOUT_V2:           ", round(mean_value_sequence_7 * 100, 2))
    print("CNN_FLAT_FC:               ", round(mean_value_sequence_6 * 100, 2))
    print("CNN_LSTM_SKIP_CAT:         ", round(mean_value_sequence_8 * 100, 2))
    print("CNN_LSTM_SKIP_MUL_FEATURE: ", round(mean_value_sequence_9 * 100, 2))
    print("CNN_LSTM_SKIP_MUL_TIME:    ", round(mean_value_sequence_10 * 100, 2))
    print("CNN_FC_LSTM_ogA:               ", round(mean_value_sequence_1_og * 100, 2))
    print("CNN_LSTM_V1_ogA:               ", round(mean_value_sequence_2_og * 100, 2))
    print("CNN_LSTM_V2_ogA:               ", round(mean_value_sequence_4_og * 100, 2))
    print("CNN_LSTM_HEAD_ogA:             ", round(mean_value_sequence_5_og * 100, 2))
    print("CNN_FC_FCOUT_V1_ogA:           ", round(mean_value_sequence_3_og * 100, 2))
    print("CNN_FC_FCOUT_V2_ogA:           ", round(mean_value_sequence_7_og * 100, 2))
    print("CNN_FLAT_FC_ogA:               ", round(mean_value_sequence_6_og * 100, 2))
    print("CNN_LSTM_SKIP_CAT_ogA:         ", round(mean_value_sequence_8_og * 100, 2))
    print("CNN_LSTM_SKIP_MUL_FEATURE_ogA: ", round(mean_value_sequence_9_og * 100, 2))
    print("CNN_LSTM_SKIP_MUL_TIME_ogA:    ", round(mean_value_sequence_10_og * 100, 2))


#list_for_switch_values_single = [item for sublist in list_for_switch_values_single for item in sublist]
#list_for_switch_values_1      = [item for sublist in list_for_switch_values_1      for item in sublist]
#list_for_switch_values_2      = [item for sublist in list_for_switch_values_2      for item in sublist]
#list_for_switch_values_3      = [item for sublist in list_for_switch_values_3      for item in sublist]
#list_for_switch_values_4      = [item for sublist in list_for_switch_values_4      for item in sublist]
#list_for_switch_values_5      = [item for sublist in list_for_switch_values_5      for item in sublist]
#list_for_switch_values_6      = [item for sublist in list_for_switch_values_6      for item in sublist]
#list_for_switch_values_7      = [item for sublist in list_for_switch_values_7      for item in sublist]
#list_for_switch_values_8      = [item for sublist in list_for_switch_values_8      for item in sublist]
#list_for_switch_values_9      = [item for sublist in list_for_switch_values_9      for item in sublist]
#list_for_switch_values_10     = [item for sublist in list_for_switch_values_10     for item in sublist]

print("\n switch averages: \n")

print("original:                  ", round(np.mean(list_for_switch_values_original) * 100, 2))
print("single-frame-based:        ", round(np.mean(list_for_switch_values_single) * 100, 2))
print("single-frame-based_ogA:        ", round(np.mean(list_for_switch_values_single_og) * 100, 2))
print("CNN_FC_LSTM:               ", round(np.mean(list_for_switch_values_1) * 100, 2))
print("CNN_LSTM_V1:               ", round(np.mean(list_for_switch_values_2) * 100, 2))
print("CNN_LSTM_V2:               ", round(np.mean(list_for_switch_values_3) * 100, 2))
print("CNN_LSTM_HEAD:             ", round(np.mean(list_for_switch_values_4) * 100, 2))
print("CNN_FC_FCOUT_V1:           ", round(np.mean(list_for_switch_values_5) * 100, 2))
print("CNN_FC_FCOUT_V2:           ", round(np.mean(list_for_switch_values_6) * 100, 2))
print("CNN_FLAT_FC:               ", round(np.mean(list_for_switch_values_7) * 100, 2))
print("CNN_LSTM_SKIP_CAT:         ", round(np.mean(list_for_switch_values_8) * 100, 2))
print("CNN_LSTM_SKIP_MUL_FEATURE: ", round(np.mean(list_for_switch_values_9) * 100, 2))
print("CNN_LSTM_SKIP_MUL_TIME:    ", round(np.mean(list_for_switch_values_10) * 100, 2))
print("CNN_FC_LSTM_ogA:               ", round(np.mean(list_for_switch_values_1_og) * 100, 2))
print("CNN_LSTM_V1_ogA:               ", round(np.mean(list_for_switch_values_2_og) * 100, 2))
print("CNN_LSTM_V2_ogA:               ", round(np.mean(list_for_switch_values_3_og) * 100, 2))
print("CNN_LSTM_HEAD_ogA:             ", round(np.mean(list_for_switch_values_4_og) * 100, 2))
print("CNN_FC_FCOUT_V1_ogA:           ", round(np.mean(list_for_switch_values_5_og) * 100, 2))
print("CNN_FC_FCOUT_V2_ogA:           ", round(np.mean(list_for_switch_values_6_og) * 100, 2))
print("CNN_FLAT_FC_ogA:               ", round(np.mean(list_for_switch_values_7_og) * 100, 2))
print("CNN_LSTM_SKIP_CAT_ogA:         ", round(np.mean(list_for_switch_values_8_og) * 100, 2))
print("CNN_LSTM_SKIP_MUL_FEATURE_ogA: ", round(np.mean(list_for_switch_values_9_og) * 100, 2))
print("CNN_LSTM_SKIP_MUL_TIME_ogA:    ", round(np.mean(list_for_switch_values_10_og) * 100, 2))

cp_list_for_switch_values_original=list_for_switch_values_original.copy()
cp_list_for_switch_values_single=list_for_switch_values_single.copy()
cp_list_for_switch_values_single_og=list_for_switch_values_single_og.copy()
cp_list_for_switch_values_1=list_for_switch_values_1.copy()
cp_list_for_switch_values_2=list_for_switch_values_2.copy()
cp_list_for_switch_values_3=list_for_switch_values_3.copy()
cp_list_for_switch_values_4=list_for_switch_values_4.copy()
cp_list_for_switch_values_5=list_for_switch_values_5.copy()
cp_list_for_switch_values_6=list_for_switch_values_6.copy()
cp_list_for_switch_values_7=list_for_switch_values_7.copy()
cp_list_for_switch_values_8=list_for_switch_values_8.copy()
cp_list_for_switch_values_9=list_for_switch_values_9.copy()
cp_list_for_switch_values_10=list_for_switch_values_10.copy()
cp_list_for_switch_values_1_og=list_for_switch_values_1_og.copy()
cp_list_for_switch_values_2_og=list_for_switch_values_2_og.copy()
cp_list_for_switch_values_3_og=list_for_switch_values_3_og.copy()
cp_list_for_switch_values_4_og=list_for_switch_values_4_og.copy()
cp_list_for_switch_values_5_og=list_for_switch_values_5_og.copy()
cp_list_for_switch_values_6_og=list_for_switch_values_6_og.copy()
cp_list_for_switch_values_7_og=list_for_switch_values_7_og.copy()
cp_list_for_switch_values_8_og=list_for_switch_values_8_og.copy()
cp_list_for_switch_values_9_og=list_for_switch_values_9_og.copy()
cp_list_for_switch_values_10_og=list_for_switch_values_10_og.copy()

# ohne zweiter szene:
del list_for_switch_values_original[1]
del list_for_switch_values_single[1]
del list_for_switch_values_single_og[1]
del list_for_switch_values_1[1]
del list_for_switch_values_2[1]
del list_for_switch_values_3[1]
del list_for_switch_values_4[1]
del list_for_switch_values_5[1]
del list_for_switch_values_6[1]
del list_for_switch_values_7[1]
del list_for_switch_values_8[1]
del list_for_switch_values_9[1]
del list_for_switch_values_10[1]
del list_for_switch_values_1_og[1]
del list_for_switch_values_2_og[1]
del list_for_switch_values_3_og[1]
del list_for_switch_values_4_og[1]
del list_for_switch_values_5_og[1]
del list_for_switch_values_6_og[1]
del list_for_switch_values_7_og[1]
del list_for_switch_values_8_og[1]
del list_for_switch_values_9_og[1]
del list_for_switch_values_10_og[1]


print("\n switch averages ohne zweiter szene: \n")

print("original:                  ", round(np.mean(list_for_switch_values_original) * 100, 2))
print("single-frame-based:        ", round(np.mean(list_for_switch_values_single) * 100, 2))
print("single-frame-based_ogA:        ", round(np.mean(list_for_switch_values_single_og) * 100, 2))
print("CNN_FC_LSTM:               ", round(np.mean(list_for_switch_values_1) * 100, 2))
print("CNN_LSTM_V1:               ", round(np.mean(list_for_switch_values_2) * 100, 2))
print("CNN_LSTM_V2:               ", round(np.mean(list_for_switch_values_3) * 100, 2))
print("CNN_LSTM_HEAD:             ", round(np.mean(list_for_switch_values_4) * 100, 2))
print("CNN_FC_FCOUT_V1:           ", round(np.mean(list_for_switch_values_5) * 100, 2))
print("CNN_FC_FCOUT_V2:           ", round(np.mean(list_for_switch_values_6) * 100, 2))
print("CNN_FLAT_FC:               ", round(np.mean(list_for_switch_values_7) * 100, 2))
print("CNN_LSTM_SKIP_CAT:         ", round(np.mean(list_for_switch_values_8) * 100, 2))
print("CNN_LSTM_SKIP_MUL_FEATURE: ", round(np.mean(list_for_switch_values_9) * 100, 2))
print("CNN_LSTM_SKIP_MUL_TIME:    ", round(np.mean(list_for_switch_values_10) * 100, 2))
print("CNN_FC_LSTM_ogA:               ", round(np.mean(list_for_switch_values_1_og) * 100, 2))
print("CNN_LSTM_V1_ogA:               ", round(np.mean(list_for_switch_values_2_og) * 100, 2))
print("CNN_LSTM_V2_ogA:               ", round(np.mean(list_for_switch_values_3_og) * 100, 2))
print("CNN_LSTM_HEAD_ogA:             ", round(np.mean(list_for_switch_values_4_og) * 100, 2))
print("CNN_FC_FCOUT_V1_ogA:           ", round(np.mean(list_for_switch_values_5_og) * 100, 2))
print("CNN_FC_FCOUT_V2_ogA:           ", round(np.mean(list_for_switch_values_6_og) * 100, 2))
print("CNN_FLAT_FC_ogA:               ", round(np.mean(list_for_switch_values_7_og) * 100, 2))
print("CNN_LSTM_SKIP_CAT_ogA:         ", round(np.mean(list_for_switch_values_8_og) * 100, 2))
print("CNN_LSTM_SKIP_MUL_FEATURE_ogA: ", round(np.mean(list_for_switch_values_9_og) * 100, 2))
print("CNN_LSTM_SKIP_MUL_TIME_ogA:    ", round(np.mean(list_for_switch_values_10_og) * 100, 2))

#ohne vierter szene:
del cp_list_for_switch_values_original[3]
del cp_list_for_switch_values_single[3]
del cp_list_for_switch_values_single_og[3]
del cp_list_for_switch_values_1[3]
del cp_list_for_switch_values_2[3]
del cp_list_for_switch_values_3[3]
del cp_list_for_switch_values_4[3]
del cp_list_for_switch_values_5[3]
del cp_list_for_switch_values_6[3]
del cp_list_for_switch_values_7[3]
del cp_list_for_switch_values_8[3]
del cp_list_for_switch_values_9[3]
del cp_list_for_switch_values_10[3]
del cp_list_for_switch_values_1_og[3]
del cp_list_for_switch_values_2_og[3]
del cp_list_for_switch_values_3_og[3]
del cp_list_for_switch_values_4_og[3]
del cp_list_for_switch_values_5_og[3]
del cp_list_for_switch_values_6_og[3]
del cp_list_for_switch_values_7_og[3]
del cp_list_for_switch_values_8_og[3]
del cp_list_for_switch_values_9_og[3]
del cp_list_for_switch_values_10_og[3]

print("\n switch averages ohne vierter szene: \n")

print("original:                  ", round(np.mean(cp_list_for_switch_values_original) * 100, 2))
print("single-frame-based:        ", round(np.mean(cp_list_for_switch_values_single) * 100, 2))
print("single-frame-based_ogA:        ", round(np.mean(cp_list_for_switch_values_single_og) * 100, 2))
print("CNN_FC_LSTM:               ", round(np.mean(cp_list_for_switch_values_1) * 100, 2))
print("CNN_LSTM_V1:               ", round(np.mean(cp_list_for_switch_values_2) * 100, 2))
print("CNN_LSTM_V2:               ", round(np.mean(cp_list_for_switch_values_3) * 100, 2))
print("CNN_LSTM_HEAD:             ", round(np.mean(cp_list_for_switch_values_4) * 100, 2))
print("CNN_FC_FCOUT_V1:           ", round(np.mean(cp_list_for_switch_values_5) * 100, 2))
print("CNN_FC_FCOUT_V2:           ", round(np.mean(cp_list_for_switch_values_6) * 100, 2))
print("CNN_FLAT_FC:               ", round(np.mean(cp_list_for_switch_values_7) * 100, 2))
print("CNN_LSTM_SKIP_CAT:         ", round(np.mean(cp_list_for_switch_values_8) * 100, 2))
print("CNN_LSTM_SKIP_MUL_FEATURE: ", round(np.mean(cp_list_for_switch_values_9) * 100, 2))
print("CNN_LSTM_SKIP_MUL_TIME:    ", round(np.mean(cp_list_for_switch_values_10) * 100, 2))
print("CNN_FC_LSTM_ogA:               ", round(np.mean(cp_list_for_switch_values_1_og) * 100, 2))
print("CNN_LSTM_V1_ogA:               ", round(np.mean(cp_list_for_switch_values_2_og) * 100, 2))
print("CNN_LSTM_V2_ogA:               ", round(np.mean(cp_list_for_switch_values_3_og) * 100, 2))
print("CNN_LSTM_HEAD_ogA:             ", round(np.mean(cp_list_for_switch_values_4_og) * 100, 2))
print("CNN_FC_FCOUT_V1_ogA:           ", round(np.mean(cp_list_for_switch_values_5_og) * 100, 2))
print("CNN_FC_FCOUT_V2_ogA:           ", round(np.mean(cp_list_for_switch_values_6_og) * 100, 2))
print("CNN_FLAT_FC_ogA:               ", round(np.mean(cp_list_for_switch_values_7_og) * 100, 2))
print("CNN_LSTM_SKIP_CAT_ogA:         ", round(np.mean(cp_list_for_switch_values_8_og) * 100, 2))
print("CNN_LSTM_SKIP_MUL_FEATURE_ogA: ", round(np.mean(cp_list_for_switch_values_9_og) * 100, 2))
print("CNN_LSTM_SKIP_MUL_TIME_ogA:    ", round(np.mean(cp_list_for_switch_values_10_og) * 100, 2))


# ohne zweiter und vierter szene:
del list_for_switch_values_original[2]
del list_for_switch_values_single[2]
del list_for_switch_values_single_og[2]
del list_for_switch_values_1[2]
del list_for_switch_values_2[2]
del list_for_switch_values_3[2]
del list_for_switch_values_4[2]
del list_for_switch_values_5[2]
del list_for_switch_values_6[2]
del list_for_switch_values_7[2]
del list_for_switch_values_8[2]
del list_for_switch_values_9[2]
del list_for_switch_values_10[2]
del list_for_switch_values_1_og[2]
del list_for_switch_values_2_og[2]
del list_for_switch_values_3_og[2]
del list_for_switch_values_4_og[2]
del list_for_switch_values_5_og[2]
del list_for_switch_values_6_og[2]
del list_for_switch_values_7_og[2]
del list_for_switch_values_8_og[2]
del list_for_switch_values_9_og[2]
del list_for_switch_values_10_og[2]

print("\n switch averages ohne zweiter und vierter szene: \n")

print("original:                  ", round(np.mean(list_for_switch_values_original) * 100, 2))
print("single-frame-based:        ", round(np.mean(list_for_switch_values_single) * 100, 2))
print("single-frame-based_ogA:        ", round(np.mean(list_for_switch_values_single_og) * 100, 2))
print("CNN_FC_LSTM:               ", round(np.mean(list_for_switch_values_1) * 100, 2))
print("CNN_LSTM_V1:               ", round(np.mean(list_for_switch_values_2) * 100, 2))
print("CNN_LSTM_V2:               ", round(np.mean(list_for_switch_values_3) * 100, 2))
print("CNN_LSTM_HEAD:             ", round(np.mean(list_for_switch_values_4) * 100, 2))
print("CNN_FC_FCOUT_V1:           ", round(np.mean(list_for_switch_values_5) * 100, 2))
print("CNN_FC_FCOUT_V2:           ", round(np.mean(list_for_switch_values_6) * 100, 2))
print("CNN_FLAT_FC:               ", round(np.mean(list_for_switch_values_7) * 100, 2))
print("CNN_LSTM_SKIP_CAT:         ", round(np.mean(list_for_switch_values_8) * 100, 2))
print("CNN_LSTM_SKIP_MUL_FEATURE: ", round(np.mean(list_for_switch_values_9) * 100, 2))
print("CNN_LSTM_SKIP_MUL_TIME:    ", round(np.mean(list_for_switch_values_10) * 100, 2))
print("CNN_FC_LSTM_ogA:               ", round(np.mean(list_for_switch_values_1_og) * 100, 2))
print("CNN_LSTM_V1_ogA:               ", round(np.mean(list_for_switch_values_2_og) * 100, 2))
print("CNN_LSTM_V2_ogA:               ", round(np.mean(list_for_switch_values_3_og) * 100, 2))
print("CNN_LSTM_HEAD_ogA:             ", round(np.mean(list_for_switch_values_4_og) * 100, 2))
print("CNN_FC_FCOUT_V1_ogA:           ", round(np.mean(list_for_switch_values_5_og) * 100, 2))
print("CNN_FC_FCOUT_V2_ogA:           ", round(np.mean(list_for_switch_values_6_og) * 100, 2))
print("CNN_FLAT_FC_ogA:               ", round(np.mean(list_for_switch_values_7_og) * 100, 2))
print("CNN_LSTM_SKIP_CAT_ogA:         ", round(np.mean(list_for_switch_values_8_og) * 100, 2))
print("CNN_LSTM_SKIP_MUL_FEATURE_ogA: ", round(np.mean(list_for_switch_values_9_og) * 100, 2))
print("CNN_LSTM_SKIP_MUL_TIME_ogA:    ", round(np.mean(list_for_switch_values_10_og) * 100, 2))