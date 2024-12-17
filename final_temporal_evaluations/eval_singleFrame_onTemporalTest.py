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

def plot_ious(sequence_based_0,
              sequence_based_1,
              sequence_based_2,
              sequence_based_3,
              sequence_num,
              means_per_seq_sequence_based_0,
              means_per_seq_sequence_based_1,
              means_per_seq_sequence_based_2,
              means_per_seq_sequence_based_3,
              highlight_area,
              ):
    if len(sequence_based_0) != len(sequence_based_1):
        raise ValueError("die beiden iou listen haben nicht dies selbe Länge!")

    x_values = range(len(sequence_based_0)) # Indizes für die x-Achse

    # plotte die Graphik
    plt.figure(figsize=(10, 6))

    # Hintergrund einfärben
    highlight_start, highlight_end = highlight_area
    plt.axvspan(highlight_start, highlight_end, color='green', alpha=0.3, label='switch')
    plt.plot(x_values, sequence_based_0, marker='.', color='blue', label=f'TEP-original                           ({means_per_seq_sequence_based_0})')
    plt.plot(x_values, sequence_based_1, marker='.', color='red',    label=f'TEP-adapted-autocrop                 ({means_per_seq_sequence_based_1})')
    plt.plot(x_values, sequence_based_2, marker='.', color='green',  label=f'single-frame-based                   ({means_per_seq_sequence_based_2})')
    plt.plot(x_values, sequence_based_3, marker='.', color='orange', label=f'single-frame-based-original-autocrop ({means_per_seq_sequence_based_3})')
    plt.title(f'Comparison between models, test sequence: {sequence_num+1}')
    plt.xlabel('Frame')
    plt.ylabel('IoU')
    plt.legend() # Legende hinzufügen
    plt.grid(True) # Gitterlinien hinzufügen
    plt.savefig(f"../imagesFilesForMasterthesis/plot_ious_sequence_{sequence_num + 1}.svg", format="svg") # speichern als svg
    print(f"saved plot_ious_sequence_{sequence_num + 1}.svg")
    plt.show()



# Einlesen der Listen aus den Textdateien
sequence_based_0 = read_list_from_file('TEP-original.txt')
sequence_based_1 = read_list_from_file('TEP-adapted-autocrop.txt')
sequence_based_2 = read_list_from_file('single-frame-based.txt')
sequence_based_3 = read_list_from_file('single-frame-based-original-autocrop.txt')

#printen der gesamten IoUs pro model
calculate_IoU_all_scenes('TEP-original',                         sequence_based_0)
calculate_IoU_all_scenes('TEP-adapted-autocrop',                 sequence_based_1)
calculate_IoU_all_scenes('single-frame-based',                   sequence_based_2)
calculate_IoU_all_scenes('single-frame-based-original-autocrop', sequence_based_3)

# Aufteilen der Listen in die Sequencen
sequence_based_0_parts, means_per_seq_sequence_based_0 = split_into_parts(sequence_based_0, 76-lösche_ersten_warmup_indices)
sequence_based_1_parts, means_per_seq_1_sequence_based = split_into_parts(sequence_based_1, 76-lösche_ersten_warmup_indices)
sequence_based_2_parts, means_per_seq_2_sequence_based = split_into_parts(sequence_based_2, 76-lösche_ersten_warmup_indices)
sequence_based_3_parts, means_per_seq_3_sequence_based = split_into_parts(sequence_based_3, 76-lösche_ersten_warmup_indices)

switch_drive_by = [(34, 75),
                   (33, 75), #(32, 65),
                   (31, 75),
                   (7, 51)]

list_for_switch_values_0 = []
list_for_switch_values_1 = []
list_for_switch_values_2 = []
list_for_switch_values_3 = []

# Für jeden Teil einen separaten Plot erstellen
for sequence_num in range(len(sequence_based_0_parts)):
    plot_ious(sequence_based_0_parts[sequence_num],
              sequence_based_1_parts[sequence_num],
              sequence_based_2_parts[sequence_num],
              sequence_based_3_parts[sequence_num],
              sequence_num,
              means_per_seq_sequence_based_0[sequence_num],
              means_per_seq_1_sequence_based[sequence_num],
              means_per_seq_2_sequence_based[sequence_num],
              means_per_seq_3_sequence_based[sequence_num],
              switch_drive_by[sequence_num],
              )
    switch_start, switch_end = switch_drive_by[sequence_num]
    mean_value_sequence_0 = np.mean(sequence_based_0_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_1 = np.mean(sequence_based_1_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_2 = np.mean(sequence_based_2_parts[sequence_num][switch_start:switch_end])
    mean_value_sequence_3 = np.mean(sequence_based_3_parts[sequence_num][switch_start:switch_end])
    list_for_switch_values_0.append(mean_value_sequence_0)
    list_for_switch_values_1.append(mean_value_sequence_1)
    list_for_switch_values_2.append(mean_value_sequence_2)
    list_for_switch_values_3.append(mean_value_sequence_3)
    
    #list_for_switch_values_single.append(sequence_based_0_parts[sequence_num][switch_start:switch_end])
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
    print("TEP-original                        : ", round(mean_value_sequence_0 * 100, 2))
    print("TEP-adapted-autocrop                : ", round(mean_value_sequence_1 * 100, 2))
    print("single-frame-based                  : ", round(mean_value_sequence_2 * 100, 2))
    print("single-frame-based-original-autocrop: ", round(mean_value_sequence_3 * 100, 2))


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

print("TEP-original                              ", round(np.mean(list_for_switch_values_0) * 100, 2))
print("TEP-adapted-autocrop                      ", round(np.mean(list_for_switch_values_1) * 100, 2))
print("single-frame-based                        ", round(np.mean(list_for_switch_values_2) * 100, 2))
print("single-frame-based-original-autocrop      ", round(np.mean(list_for_switch_values_3) * 100, 2))