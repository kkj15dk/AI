import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
from DNAconversion import one_hot_encode

# Set a random seed in a bunch of different places
def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The random seed to set.

    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set as {seed}")

# Define the dataset
def quick_split(df, split_frac=0.8):
    '''
    Given a df of samples, randomly split indices between
    train and test at the desired fraction
    
    Parameters:
        df (DataFrame): The input DataFrame containing the samples.
        split_frac (float): The fraction of indices to be allocated for training.
    
    Returns:
        tuple: A tuple containing two DataFrames, the training set and the test set.
    '''
    cols = df.columns # original columns, use to clean up reindexed cols
    df = df.reset_index()

    # shuffle indices
    idxs = list(range(df.shape[0]))
    random.shuffle(idxs)

    # split shuffled index list by split_frac
    split = int(len(idxs)*split_frac)
    train_idxs = idxs[:split]
    test_idxs = idxs[split:]
    
    # split dfs and return
    train_df = df[df.index.isin(train_idxs)]
    test_df = df[df.index.isin(test_idxs)]
        
    return train_df[cols], test_df[cols]

class SeqDatasetOHE(Dataset):
    '''
    Dataset for one-hot-encoded sequences
    
    Args:
        df (pandas.DataFrame): The input dataframe containing the sequences and targets.
        seq_col (str): The column name in the dataframe that contains the sequences. Default is 'seq'.
        target_col (str): The column name in the dataframe that contains the target targets. Default is 'score'.
    '''
    def __init__(self,
                 df,
                 seq_col='seq',
                 target_col='score'
                ):
        
        #  Get the X examples
        # extract the DNA from the appropriate column in the df
        self.seqs = list(df[seq_col].values)
        # self.seq_len = len(self.seqs[0])
        
        # one-hot encode sequences
        self.ohe_seqs = [torch.tensor(one_hot_encode(x)) for x in self.seqs]
    
        # Get the Y targets
        self.targets = torch.tensor(list(df[target_col].values)).unsqueeze(1)
        
    def __len__(self): 
        '''
        Returns the total number of samples in the dataset.
        
        Returns:
            int: The number of samples in the dataset.
        '''
        return len(self.seqs)
    
    def __getitem__(self,idx):
        '''
        Returns the input and output for a given index.
        
        Args:
            idx (int): The index of the sample to retrieve.
        
        Returns:
            tuple: A tuple containing the one-hot encoded sequence (X) and the target (Y).
        '''
        seq = self.ohe_seqs[idx]
        label = self.targets[idx]
        
        return seq, label

def collate_fn(batch):
    '''
    Given a batch of data, collate the inputs and outputs into their own tensors.
    
    Args:
        batch (list): A list of tuples containing the input and ouput values.
    
    Returns:
        tuple: A tuple containing the input and output tensors.
    '''
    # separate the X and Y values
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # stack the inputs into a single tensor
    inputs = pad_sequence(inputs, batch_first=True)
    
    # stack the targets into a single tensor
    targets = torch.stack(targets)
    
    return inputs, targets

def build_dataloaders(train_df,
                      test_df,
                      seq_col='seq',
                      target_col='score',
                      batch_size=1,
                      collate_fn=collate_fn,
                      shuffle=True
                     ):
    '''
    Given a train and test df with some batch construction
    details, put them into custom SeqDatasetOHE() objects. 
    Give the Datasets to the DataLoaders and return.

    Parameters:
    train_df (DataFrame): The training dataframe.
    test_df (DataFrame): The testing dataframe.
    seq_col (str): The name of the column containing the sequences. Default is 'seq'.
    target_col (str): The name of the column containing the target values. Default is 'score'.
    batch_size (int): The batch size for the dataloaders. Default is 1.
    shuffle (bool): Whether to shuffle the data during training. Default is True.

    Returns:
    train_dl (DataLoader): The dataloader for the training dataset.
    test_dl (DataLoader): The dataloader for the testing dataset.
    '''
    
    # create Datasets    
    train_ds = SeqDatasetOHE(train_df,seq_col=seq_col,target_col=target_col)
    test_ds = SeqDatasetOHE(test_df,seq_col=seq_col,target_col=target_col)

    # Put DataSets into DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    return train_dl,test_dl

class TemporalPyramidPooling(nn.Module):
    """
    Temporal Pyramid Pooling module.

    This module performs temporal pyramid pooling on the input tensor.
    It divides the input tensor into multiple temporal levels and applies adaptive average pooling
    to each level. The resulting pooled features are then concatenated and returned as the output.

    Args:
        pyramid_levels (list): A list of integers specifying the temporal levels for pooling.

    Attributes:
        pyramid_levels (list): The temporal levels for pooling.

    """

    def __init__(self, pyramid_levels):
        super(TemporalPyramidPooling, self).__init__()
        self.pyramid_levels = pyramid_levels

    def forward(self, x):
        B, C, T = x.size()
        pyramid_outputs = []
        for pyramid_level in self.pyramid_levels:
            tpp = nn.AdaptiveAvgPool1d(pyramid_level)
            pooled = tpp(x)
            flattened = pooled.view(B, -1)
            pyramid_outputs.append(flattened)
        output = torch.cat(pyramid_outputs, dim=1)
        return output

class MaxPooling(nn.Module):
    def __init__(self, kernel_size):
        """
        Initializes a MaxPooling module.

        Args:
            kernel_size (int): The size of the max pooling window.

        """
        super(MaxPooling, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size, stride=None, padding=0,
                                    dilation=1, return_indices=False, 
                                    ceil_mode=False)

    def forward(self, x):
        """
        Performs forward pass through the MaxPooling module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after max pooling.

        """
        x = self.maxpool(x)
        return x

def train_loop(n, DEVICE, train_dl, model, loss_fn, optimizer, train_running_loss):
    """
    Trains the model for one epoch using the provided training data.

    Args:
        n (int): The number of batches after which to perform optimization.
        DEVICE (torch.device): The device to use for training (e.g., 'cuda' or 'cpu').
        train_dl (torch.utils.data.Dataloader): The training data loader.
        model (torch.nn.Module): The model to train.
        loss_fn: The loss function.
        optimizer: The optimizer.
        train_running_loss (float): The running loss for the training data.

    Returns:
        float: The average loss for the epoch.
    """
    for i, (inputs, targets) in enumerate(train_dl):
        # Move the inputs and targets to the device
        inputs = inputs.transpose(1,2).float().to(DEVICE)
        targets = targets.squeeze(1).float().to(DEVICE)
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()

        # Perform optimization every n batches
        if (i + 1) % n == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_running_loss += loss.item()
    
    # Perform optimization for remaining batches if total number of batches is not a multiple of n
    if len(train_dl) % n != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Calculate the average loss for the epoch
    train_avg_loss = train_running_loss / len(train_dl)
    return train_avg_loss

def val_loop(DEVICE, val_dl, model, loss_fn, val_running_loss, all_preds, all_targets, early_stopping_epochs=5):
    """
    Perform the validation loop for the given model.

    Args:
        DEVICE (torch.device): The device to run the computations on.
        val_dl (torch.utils.data.DataLoader): The validation dataloader.
        model (torch.nn.Module): The model to evaluate.
        loss_fn: The loss function used for evaluation.
        val_running_loss (float): The running loss for validation.
        all_preds (list): List to store all the predictions.
        all_targets (list): List to store all the targets.

    Returns:
        float: The average validation loss.
        list: All the predictions.
        list: All the targets.
    """

    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for inputs, targets in val_dl: # iterate over the validation dataloader
        inputs = inputs.transpose(1,2).float().to(DEVICE)
        targets = targets.squeeze(1).float().to(DEVICE)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Get predictions
        preds = torch.sigmoid(outputs) > 0.5 # threshold the output to get the class prediction
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        # Calculate loss
        val_running_loss += loss.item()
        val_avg_losses = val_running_loss / len(val_dl)
        if val_running_loss < best_val_loss:
            best_val_loss = val_running_loss
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement == early_stopping_epochs:
                print('Early stopping')
                break
    return val_avg_losses, all_preds, all_targets