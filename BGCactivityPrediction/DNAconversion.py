from Bio import SeqIO
import numpy as np
import os
import random
import torch.nn.functional as F

def one_hot_encode(seq, aa=False):
    """
    Given a DNA or AA sequence, return its one-hot encoding
    """
    # Make sure seq has only allowed bases
    if aa:
        allowed = set("ACDEFGHIKLMNPQRSTVWY><-X")
    else:
        allowed = set("ACTGN")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(f"Sequence contains chars not in allowed alphabet (ACGTN, or aa): {invalid}")
        
        # Dictionary returning one-hot encoding for each nucleotide or amino acid
    if aa:
        dictionary = {
            'A': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'C': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'D': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'E': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'F': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'G': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'H': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'I': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'K': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'L': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'M': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'N': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'P': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'Q': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'R': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'S': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'T': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'V': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'W': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            'Y': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            '>': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            '<': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            '-': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            'X': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]#,
            # 'B': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            # 'J': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        }
    else:
        dictionary = {'A':[1.0,0.0,0.0,0.0],
                      'C':[0.0,1.0,0.0,0.0],
                      'G':[0.0,0.0,1.0,0.0],
                      'T':[0.0,0.0,0.0,1.0],
                      'N':[0.0,0.0,0.0,0.0]}
    
    # Create array from nucleotide sequence
    vec = np.array([dictionary[x] for x in seq])
    vec = vec.T # Transpose the array to make format match the expected input for a CNN

    return vec

def hot_one_encode(onehotencoded, aa=False):
    """
    Given a one-hot encoded sequence, return the original sequence
    """
    # Dictionary returning nucleotide or amino acid for each one-hot encoding
    if aa:
        dictionary = {
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'A',
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'C',
            (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'D',
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'E',
            (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'F',
            (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'G',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'H',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'I',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'K',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'L',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'M',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'N',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'P',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'Q',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'R',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'S',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'T',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'V',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0): 'W',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0): 'Y',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0): '>',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0): '<',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0): '-',
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 'X'
        }
    else:
        dictionary = {
            (1.0, 0.0, 0.0, 0.0): 'A',
            (0.0, 1.0, 0.0, 0.0): 'C',
            (0.0, 0.0, 1.0, 0.0): 'G',
            (0.0, 0.0, 0.0, 1.0): 'T',
            (0.0, 0.0, 0.0, 0.0): 'N'
        }
    
    # Create array from one-hot encoded sequence
    seq = ''.join([dictionary[tuple(x)] for x in onehotencoded.T])
    
    return seq

def filter_fasta(input_file, output_file, min_length=None, max_length=None):
    with open(input_file, "r") as input_handle, open(output_file, "w") as output_handle:
        sequences = SeqIO.parse(input_handle, "fasta")
        if min_length is not None:
            sequences = (record for record in sequences if len(record.seq) > min_length)
        if max_length is not None:
            sequences = (record for record in sequences if len(record.seq) < max_length)

        SeqIO.write(sequences, output_handle, "fasta")

def pad_to_length(tensor, length, padding_value=0):
    """
    Pads or truncates a tensor to a specified length with a given padding value.

    Args:
        tensor (torch.Tensor): The input tensor to be padded.
        length (int): The desired length of the tensor after padding.
        padding_value (int, optional): The value used for padding. Defaults to 0.

    Returns:
        tuple: A tuple containing the padded tensor and a mask tensor indicating the padded regions.
    """
    if tensor.shape[1] < length:
        tensor = F.pad(tensor, (0, length - tensor.shape[1]), value=padding_value)
    else:
        tensor = tensor[:, :length]
    return tensor

def pad_string_right(string, length, padding_value='-'):
    """
    Pads or truncates a string to a specified length with a given padding value.

    Args:
        string (str): The input string to be padded.
        length (int): The desired length of the string after padding.
        padding_value (str, optional): The character used for padding. Defaults to '-'.

    Returns:
        str: The padded string.
    """
    if len(string) < length:
        string = string.ljust(length, padding_value)
    else:
        string = string[:length]
    return string

def pad_string(string, length, padding_value='-'):
    """
    Pads or truncates a string to a specified length with a given padding value.

    Args:
        string (str): The input string to be padded.
        length (int): The desired length of the string after padding.
        padding_value (str, optional): The character used for padding. Defaults to '-'.

    Returns:
        str: The padded string.
    """
    if len(string) < length:
        rand_len = random.randint(len(string), length)
        left_pad = rand_len - len(string)
        right_pad = length - rand_len
        string = padding_value * left_pad + string + padding_value * right_pad
    else:
        string = string[:length]
    return string


# Description: Generate random amino acid sequences for testing
def random_aa_seq(n):
    lsseq = []
    for i in range(n):
        # Generate a random aa sequence
        seq = "M"
        for j in range(3):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY-")
        seq += "HINQA"
        seq += random.choice(["----","ACDE"])
        for j in range(2):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY-")
        seq += random.choice(["----","FGHI"])
        for j in range(2):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY-")
        seq += random.choice(["----","KLMN"])
        for j in range(3):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY-")
        seq += random.choice(["----", "PQRS"])
        for j in range(4):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY-")
        seq += random.choice(["----", "TVWY"])
        # Print the sequence
        # print(seq)
        lsseq.append(seq)
    return lsseq

def random_aa_seq_unaligned(n):
    lsseq = []
    for i in range(n):
        # Generate a random aa sequence
        seq = "M"
        for j in range(3):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY")
        seq += "HINQA"
        seq += random.choice(["","ACDE"])
        for j in range(2):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY")
        seq += random.choice(["","FGHI"])
        for j in range(2):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY")
        seq += random.choice(["","KLMN"])
        for j in range(3):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY")
        seq += random.choice(["", "PQRS"])
        for j in range(4):
            seq += random.choice("ACDEFGHIKLMNPQRSTVWY")
        seq += random.choice(["", "TVWY"])
        # Print the sequence
        # print(seq)
        lsseq.append(seq)
    return lsseq


# Usage
if __name__ == "__main__":
    input_file = "new4_PKSs.fa"
    output_file = "test_PKSs.fa"

    if not os.path.exists(output_file):
        filter_fasta(input_file, output_file, 0, 400000)

    # Read the sequences from the input file
    sequences = SeqIO.parse(input_file, "fasta")
    sorted_sequences = sorted(sequences, key=lambda x: len(x.seq))
    # Get the 5 shortest and longest sequences
    shortest_sequences = sorted_sequences[:5]
    longest_sequences = sorted_sequences[-5:-1]

    print("there are", len(sorted_sequences), "sequences in the file")

    if len(sorted_sequences) != len(set(str(record.seq) for record in sorted_sequences)):
        i = len(sorted_sequences) - len(set(str(record.seq) for record in sorted_sequences))
        print("There are " + str(i) + " duplicate sequences in the file")
    else:
        print("There are no duplicate sequences in the file")

    # Print the length and accession number of the shortest sequences
    print("Shortest sequences:")
    for seq in shortest_sequences:
        print(f"Length: {len(seq.seq)}, Accession: {seq.description}")

    # Print the length and accession number of the longest sequences
    print("Longest sequences:")
    for seq in longest_sequences:
        print(f"Length: {len(seq.seq)}, Accession: {seq.description}")

