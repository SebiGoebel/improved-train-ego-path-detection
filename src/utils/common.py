import logging
import random

import numpy as np
import torch
from torchvision.transforms import v2 as transforms

# for splitting temporal dataset
import os
from collections import defaultdict

to_scaled_tensor = transforms.Compose(
    [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
)  # [0, 255] PIL.Image or numpy.ndarray to [0, 1] torchvision Image (torch.Tensor)


def split_dataset(indices, proportions):
    """Splits the dataset indices into training, validation and test sets.

    Args:
        indices (list): List of indices of the dataset.
        proportions (tuple): Tuple containing the proportions of the training, validation and test sets (e.g. (0.8, 0.1, 0.1)).

    Returns:
        tuple: Tuple containing the training, validation and test sets indices.
    """
    train_prop, val_prop, test_prop = proportions
    train_indices = indices[: int(train_prop * len(indices))]
    val_indices = indices[
        int(train_prop * len(indices)) : int((train_prop + val_prop) * len(indices))
    ]
    test_indices = indices[
        int((train_prop + val_prop) * len(indices)) : int(
            (train_prop + val_prop + test_prop) * len(indices)
        )
    ]

    # Write to fixed text files
    with open('dataset_indices/train.txt', 'w') as f:
        for idx in train_indices:
            f.write(str(idx) + '\n')

    with open('dataset_indices/val.txt', 'w') as f:
        for idx in val_indices:
            f.write(str(idx) + '\n')

    with open('dataset_indices/test.txt', 'w') as f:
        for idx in test_indices:
            f.write(str(idx) + '\n')
    
    return train_indices, val_indices, test_indices

def split_dataset_by_sequence(labels_path, proportions):
    """Splits the dataset sequences into training, validation, and test sets.

    Args:
        labels_path (str): The labels_path containing the annotation file. (e.g. "datasets/temporalSwitchDataset_TEPForamt/labels/temporalLabels.json")
        proportions (tuple): Tuple containing the number of sequences for the training, validation, and test sets (e.g. (30, 4, 4)).

    Returns:
        tuple: Tuple containing the training, validation and test sets indices.
    """
    train_count, val_count, test_count = proportions

    # Step 1: Read all files from the labels_path and group by sequence
    sequences = defaultdict(list)
    for filename in os.listdir(labels_path):
        if filename.endswith('.png'):
            sequence_id = filename.split('_frame_')[0]
            sequences[sequence_id].append(filename)
    
    # Convert sequences from dict to list of lists and sort each sequence
    sequences = [sorted(seq) for seq in sequences.values()]

    if train_count + val_count + test_count != len(sequences):
        raise ValueError("The sum of proportions must equal the number of sequences.")

    # Step 2: Split the sequences into training, validation, and test sets
    train_sequences = sequences[:train_count]
    val_sequences = sequences[train_count:train_count + val_count]
    test_sequences = sequences[train_count + val_count:train_count + val_count + test_count]

    # Flatten the lists of lists into single lists of indices
    all_filenames = [filename for seq in sequences for filename in seq]
    #filename_to_index = {filename: idx + 1 for idx, filename in enumerate(all_filenames)}
    filename_to_index = {filename: idx for idx, filename in enumerate(all_filenames)}

    train_indices = [filename_to_index[filename] for seq in train_sequences for filename in seq]
    val_indices = [filename_to_index[filename] for seq in val_sequences for filename in seq]
    test_indices = [filename_to_index[filename] for seq in test_sequences for filename in seq]

    # Write to fixed text files
    with open('dataset_indices_temporal/temporal_train.txt', 'w') as f:
        for idx in train_indices:
            f.write(str(idx) + '\n')

    with open('dataset_indices_temporal/temporal_val.txt', 'w') as f:
        for idx in val_indices:
            f.write(str(idx) + '\n')

    with open('dataset_indices_temporal/temporal_test.txt', 'w') as f:
        for idx in test_indices:
            f.write(str(idx) + '\n')

    return train_indices, val_indices, test_indices

# Beispielaufruf der Funktion
#labels_path = 'images'
#proportions = (30, 4, 4)
#train_indices, val_indices, test_indices = split_dataset_by_sequence(labels_path, proportions)

def split_dataset_by_sequence_from_lists(labels_path, train_sequence_indices, val_sequence_indices, test_sequence_indices):
    """Splits the dataset sequences into training, validation, and test sets using provided lists of indices.

    Args:
        labels_path (str): The labels_path containing the annotation file.
        train_sequence_indices (list): List of indices for training sequences.
        val_sequence_indices (list): List of indices for validation sequences.
        test_sequence_indices (list): List of indices for test sequences.

    Returns:
        tuple: Tuple containing the training, validation, and test sets indices.
    """

    # Step 1: Read all files from the labels_path and group by sequence
    sequences = defaultdict(list)
    for filename in os.listdir(labels_path):
        if filename.endswith('.png'):
            sequence_id = filename.split('_frame_')[0]
            sequences[sequence_id].append(filename)
    
    # Convert sequences from dict to list of lists and sort each sequence
    sequences = [sorted(seq) for seq in sequences.values()]

    # Print Sequences
    for seq_index, seq in enumerate(sequences):
        if seq:  # Überprüfen, ob die Sequenz nicht leer ist
            print(f"Sequence {seq_index}: {seq[0]} to {seq[-1]}")

    # Step 2: Select the sequences based on the provided indices
    train_sequences = [sequences[i] for i in train_sequence_indices]
    val_sequences = [sequences[i] for i in val_sequence_indices]
    test_sequences = [sequences[i] for i in test_sequence_indices]

    # Step 3: Flatten the lists of lists into single lists of indices
    all_filenames = [filename for seq in sequences for filename in seq]
    filename_to_index = {filename: idx for idx, filename in enumerate(all_filenames)}

    train_indices = [filename_to_index[filename] for seq in train_sequences for filename in seq]
    val_indices = [filename_to_index[filename] for seq in val_sequences for filename in seq]
    test_indices = [filename_to_index[filename] for seq in test_sequences for filename in seq]

    # Step 4: Write to fixed text files
    with open('dataset_indices_temporal/temporal_train.txt', 'w') as f:
        for idx in train_indices:
            f.write(str(idx) + '\n')

    with open('dataset_indices_temporal/temporal_val.txt', 'w') as f:
        for idx in val_indices:
            f.write(str(idx) + '\n')

    with open('dataset_indices_temporal/temporal_test.txt', 'w') as f:
        for idx in test_indices:
            f.write(str(idx) + '\n')

    return train_indices, val_indices, test_indices


def simple_logger(name, level, terminator="\n"):
    """Creates a simple logger to print messages to the console with no additional information.

    Args:
        name (str): Identifier of the logger.
        level (str): Level of the logger. Only messages with this level or higher will be printed.
        terminator (str, optional): String to append to the end of each message. Defaults to "\n".

    Returns:
        logging.Logger: Simple logger.
    """
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logger = logging.getLogger(name)
    logger.setLevel(levels[level])
    handler = logging.StreamHandler()
    handler.setLevel(levels[level])
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.terminator = terminator
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_worker_seeds(worker_id):
    seed = torch.initial_seed() % 2**32  # clamp to 32-bit
    random.seed(seed)
    np.random.seed(seed)
