import unittest
import torch
from BGCactivityPrediction.main import DNA_CNN, train_loop

class TestDNACNN(unittest.TestCase):
    def setUp(self):
        self.model = DNA_CNN()
        self.input = torch.randn(32, 4, 100)  # A batch of 32 sequences, each with 1 channel and 100 time steps

    def test_forward(self):
        output = self.model(self.input)
        self.assertEqual(output.shape, (32, 4))  # The output should have shape (batch_size, output_size)

    def test_train_loop(self):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dl = torch.utils.data.DataLoader(...)  # Replace ... with your training data loader
        model = DNA_CNN()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        train_running_loss = 0.0

        train_avg_loss = train_loop(10, DEVICE, train_dl, model, criterion, optimizer, train_running_loss)
        self.assertIsInstance(train_avg_loss, float)  # The average loss should be a float

if __name__ == '__main__':
    unittest.main()