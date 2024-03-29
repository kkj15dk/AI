import selfies as sf
import random

def smile_to_selfie(smile):
    return sf.encoder(smile)

def selfie_to_smile(selfie):
    return sf.decoder(selfie)

def get_alphabet_from_dataset(selfies):
    alphabet = sf.get_alphabet_from_selfies(selfies)
    alphabet.add("[nop]")
    alphabet = list(sorted(alphabet))

    pad_to_len = max(sf.len_selfies(selfie) for selfie in selfies)
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

    return alphabet, pad_to_len, symbol_to_idx

def one_hot_encoder(selfie, symbol_to_idx, pad_to_len):
    onehot = sf.selfies_to_encoding(
        selfies=selfie,
        vocab_stoi=symbol_to_idx,
        pad_to_len=pad_to_len,
        enc_type="one_hot",
    )
    return onehot

class SMILES_one_hot_encoder:
    '''
    Use on a dataset of SMILES strings to convert them to SELFIES strings.
    '''
    def __init__(self, smiles):
        self.smiles = smiles
        self.selfies = [smile_to_selfie(smile) for smile in self.smiles]
        self.alphabet, self.pad_to_len, self.symbol_to_idx = get_alphabet_from_dataset(self.selfies)

    def one_hot_encode_dataset(self):
        dataset_one_hot_encoded = [one_hot_encoder(selfie, self.symbol_to_idx, self.pad_to_len) for selfie in self.selfies]
        return dataset_one_hot_encoded

if __name__ == "__main__":
    smiles_dataset = ["CCO", "CCN", "CCOC", "CCNC"]
    converter = SMILES_one_hot_encoder(smiles_dataset)
    print(converter.one_hot_encode_dataset())
