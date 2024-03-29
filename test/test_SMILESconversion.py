from BGCactivityPrediction.SMILESconversion import *
import unittest

class TestSMILESOneHotEncoder(unittest.TestCase):
    def setUp(self):
        self.smiles = ["CCO", "CCN", "CCOC", "CCNC"]
        self.encoder = SMILES_one_hot_encoder(self.smiles)

    def test_smile_to_selfie(self):
        selfie = smile_to_selfie('CCO')
        self.assertEqual(selfie, '[C][C][O]')

    def test_selfie_to_smile(self):
        smile = selfie_to_smile('[C][C][O]')
        self.assertEqual(smile, 'CCO')

    def test_get_alphabet_from_dataset(self):
        self.assertEqual(self.encoder.alphabet, ['[C]', '[N]', '[O]', '[nop]'])
        self.assertEqual(self.encoder.pad_to_len, 4)
        self.assertEqual(self.encoder.symbol_to_idx, {'[C]': 0, '[N]': 1, '[O]': 2, '[nop]': 3})

    def test_one_hot_encoder(self):
        onehot = one_hot_encoder('[C][C][O]', self.encoder.symbol_to_idx, self.encoder.pad_to_len)
        self.assertEqual(onehot, [[1, 0, 0, 0],[1, 0, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])

    def test_one_hot_encode_dataset(self):
        dataset_one_hot_encoded = self.encoder.one_hot_encode_dataset()
        expected_output = [[[1, 0, 0, 0],[1, 0, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], 
                           [[1, 0, 0, 0],[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1]], 
                           [[1, 0, 0, 0],[1, 0, 0, 0],[0, 0, 1, 0],[1, 0, 0, 0]],
                           [[1, 0, 0, 0],[1, 0, 0, 0],[0, 1, 0, 0],[1, 0, 0, 0]]]
                        
        self.assertEqual(dataset_one_hot_encoded, expected_output)

if __name__ == '__main__':
    unittest.main()