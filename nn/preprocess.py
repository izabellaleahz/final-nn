# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """

    pos_seqs = [s for s, l in zip(seqs, labels) if l]
    neg_seqs = [s for s, l in zip(seqs, labels) if not l]

    # oversample the minority class to match the majority class
    if len(pos_seqs) < len(neg_seqs):
        minority, majority = pos_seqs, neg_seqs
        min_label, maj_label = True, False
    else:
        minority, majority = neg_seqs, pos_seqs
        min_label, maj_label = False, True

    # sample with replacement from minority to match majority size
    oversampled = list(np.random.choice(minority, size=len(majority), replace=True))

    sampled_seqs = majority + oversampled
    sampled_labels = [maj_label] * len(majority) + [min_label] * len(oversampled)

    return sampled_seqs, sampled_labels


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """

    encoding_map = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }

    encodings = []
    for seq in seq_arr:
        encoded = []
        for base in seq.upper():
            encoded.extend(encoding_map.get(base, [0, 0, 0, 0]))
        encodings.append(encoded)

    return np.array(encodings)
