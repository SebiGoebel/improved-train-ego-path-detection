def lösche_warmup_indices(liste, löschen=29, schritt=76):
    """
    löscht die ersten paar indices aus einer sequence herraus um einen fairen Vergleich zeischen single-fram-based model und LSTM model zu garantieren.
    jede GT ist damit gleich.
    """
    ergebnis = []
    for i in range(0, len(liste), schritt):
        ergebnis.extend(liste[i+löschen:i+schritt])
    return ergebnis

# datei einlesen
with open("calculateIoU_singleFrame_video_ious_kind-donkey-84_newDataset_ganzesDataset_0_frames_gelöscht.txt", "r") as file:
    numbers = [float(line.strip()) for line in file.readlines()]

numbers = lösche_warmup_indices(numbers)

# Liste in eine neue Datei (test_out.txt) schreiben
with open("calculateIoU_singleFrame_video_ious_kind-donkey-84_newDataset_ganzesDataset_29_frames_gelöscht.txt", "w") as output_file:
    for number in numbers:
        output_file.write(f"{number}\n")