import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from DNAconversion import *
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from MLvisualisation import *
from MLbuilding import *

# Set the number of epochs and the learning rate
num_epochs = 30
lr = 0.1
n = 1  # number of batches to accumulate gradients over
kernel_size = 1
maxpool_size = 2
# pyramid_levels = [4]
input_channels = ['A', 'C', 'G','T']
output_channels = ['C1', 'C2', 'C3', 'C4']
num_kernels = 16
batch_size = 100
a = 20
seq_len = b = 100
n = 2000
    
set_seed(15)

# set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device set as {DEVICE}")

# make some random data
def abmers(a,b,n=100):
    '''Generate a list of n dna seqs with random length from a to b'''
    seqs = []
    for _ in range(n):
        length = random.randint(a,b)
        seq = ''.join(random.choices('ACGT', k=length))
        seqs.append(seq)
    return seqs

# Define the dataset
seqs = abmers(a, b, n)

data = []
for seq in seqs:
    score = [0 for i in range(4)]
    if 'ATA' in seq:
        score[0] = 1
    if 'CGC' in seq:
        score[1] = 1
    if 'GTG' in seq:
        score[2] = 1
    if 'TCT' in seq:
        score[3] = 1
    data.append((seq,score))
    df = pd.DataFrame(data, columns=['seq','score'])

full_train_df, test_df = quick_split(df)
train_df, val_df = quick_split(full_train_df)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

print(seqs[0])
# print(train_df)

full_train_dl, val_dl = build_dataloaders(full_train_df, val_df, batch_size=batch_size)
train_dl, test_dl = build_dataloaders(train_df, test_df, batch_size=batch_size)

class DNA_CNN(nn.Module):
    def __init__(self, input_channels, num_kernels, kernel_size, maxpool_size, seq_len, output_channels):
        super(DNA_CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, num_kernels, kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(maxpool_size),
            # nn.PyramidPooling(pyramid_levels),
            nn.Flatten(),
            nn.Linear(num_kernels*(seq_len-kernel_size+1)//maxpool_size, output_channels)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

# Instantiate the model
model = DNA_CNN(len(input_channels), 
                num_kernels, 
                kernel_size, 
                maxpool_size, 
                seq_len, 
                len(output_channels)
                ).to(DEVICE)

# Define the loss function
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)

optimizer.zero_grad()  # initialize gradients

# Initialize variables
all_preds = []
all_labels = []
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()
    
    # Initialize the running loss
    train_running_loss = 0.0
    val_running_loss = 0.0

    # Training loop
    train_avg_loss = train_loop(n, DEVICE, train_dl, model, criterion, optimizer, train_running_loss)
    train_losses.append(train_avg_loss)

    # Evaluation loop
    model.eval()  # put model in evaluation mode
    with torch.no_grad():  # disable gradient computation
        val_avg_losses, all_preds, all_labels = val_loop(DEVICE, val_dl, model, criterion, val_running_loss, all_preds, all_labels)
        val_losses.append(val_avg_losses)
    # Print the average loss for the epoch
    val_acc = np.mean(np.array(all_preds).flatten() == np.array(all_labels).flatten())
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_losses:.4f} | Val Acc: {val_acc:.4f}")

# Generate confusion matrix
cm = multilabel_confusion_matrix(all_labels, all_preds)

# Plot confusion matrix for each class
for i, matrix in enumerate(cm):
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
    ax.set_title(f'Class {i+1}')
    plt.savefig(f'confusion_matrix_class_{i+1}.png')  # save figure to file

cnn_data_label = (train_losses, val_losses, "CNN")
quick_loss_plot([cnn_data_label])

conv_layers, model_weights, bias_weights = get_conv_layers_from_model(model)
view_filters(model_weights)

# just use some seqs from test_df to activate filters
seqs = test_df['seq'].values
some_seqs = random.choices(seqs, k=2000)

filter_activations = get_filter_activations(some_seqs, conv_layers[0], DEVICE)
view_filters_and_logos(model_weights,filter_activations)