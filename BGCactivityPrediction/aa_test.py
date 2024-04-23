import random

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
        seq += "*"
        # Print the sequence
        # print(seq)
        lsseq.append(seq)
    return lsseq

if __name__ == "__main__":
    random_aa_seq(10)