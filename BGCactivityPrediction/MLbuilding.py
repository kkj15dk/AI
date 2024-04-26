import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
from DNAconversion import one_hot_encode, hot_one_encode
from Bio import SeqIO

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

class SeqDatasetOHE(Dataset):
    '''
    Dataset for one-hot-encoded sequences
    
    Args:
        df (pandas.DataFrame): The input dataframe containing the sequences and optionally the targets. Assumes targets are provided.
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
        self.ohe_seqs = [torch.tensor(one_hot_encode(seq)) for seq in self.seqs]
    
        # Get the targets
        self.targets = None
        if target_col:
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
        if self.targets is not None:
            label = self.targets[idx]
            return seq, label
        else:
            return seq

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

class Encoder_2(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim, kernel_size, stride, padding, output_len):
        super(Encoder_2, self).__init__()
        self.output_len = output_len
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Conv1d(hidden_channels*2, hidden_channels*4, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            # nn.Linear(hidden_channels * 2 * self.output_len, hidden_channels * 2 * self.output_len),
            # nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_channels * 2 * self.output_len, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels * 2 * self.output_len, latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder_2(nn.Module):
    def __init__(self, hidden_channels, input_channels, latent_dim, kernel_size, stride, padding, output_len):
        super(Decoder_2, self).__init__()
        self.hidden_channels = hidden_channels
        self.output_len = output_len
        self.fc_z = nn.Linear(latent_dim, hidden_channels * 2 * self.output_len)
        # self.fc_z = nn.Linear(latent_dim, hidden_channels * 4 * self.output_len)
        self.decoder = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.ConvTranspose1d(hidden_channels*4, hidden_channels*2, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(hidden_channels*2, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(hidden_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Softmax(dim = 1)
        )

    def forward(self, z):
        z = self.fc_z(z)
        z = z.view(-1, self.hidden_channels * 2, self.output_len)
        x_hat = self.decoder(z)
        return x_hat

class cVAE_2(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim, kernel_size, stride, padding, max_len):
        super(cVAE_2, self).__init__()

        # Define the output lengths between different layers of the model. hopefully this will make the model easier to manipulate later on
        self.max_len = max_len
        self.output_len = int(((self.max_len - kernel_size + 2*padding) / stride + 1) ) # Can give weird values if stride doesn't divide the length
        print("Output length: ", self.output_len)
        # Encoder
        self.encoder = Encoder_2(input_channels, hidden_channels, latent_dim, kernel_size, stride, padding, self.output_len)
        # Decoder
        self.decoder = Decoder_2(hidden_channels, input_channels, latent_dim, kernel_size, stride, padding, self.output_len)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class Argmax(nn.Module):
    def __init__(self, dim):
        super(Argmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.argmax(dim = self.dim)

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim, kernel_size, stride, padding, layers, pooling, max_len, pooling_window, embedding, embedding_dim, pool_doublingtime, conv_doublingtime, pooling_method, inner_dim):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.enc_ref = []
        self.max_len = max_len
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        if embedding:
            self.enc_ref.append(Argmax(1))
            self.enc_ref.append(nn.Embedding(self.input_channels, embedding_dim))
            self.enc_ref.append(Permute(0, 2, 1))
            self.input_channels = embedding_dim
        for i in range(layers):
            self.enc_ref.append(nn.Conv1d(self.input_channels, self.hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            self.enc_ref.append(nn.LeakyReLU())
            self.max_len = int(((self.max_len - kernel_size + 2*padding) / stride + 1) ) # Can give weird values if stride doesn't divide the length
            self.input_channels = self.hidden_channels
            if (i+1) % pool_doublingtime == 0: # Sorry for the nesting...
                if pooling:
                    self.max_len = int(np.ceil(self.max_len/pooling_window))
                    if pooling_method == 'max':
                        self.enc_ref.append(nn.AdaptiveMaxPool1d(self.max_len))
                    elif pooling_method == 'avg':
                        self.enc_ref.append(nn.AdaptiveAvgPool1d(self.max_len))
            if (i+1) % conv_doublingtime == 0:
                self.hidden_channels *= 2
        self.encoder = nn.Sequential(
            *self.enc_ref,
            nn.Flatten()
        )

        
        self.fc_0 = nn.Linear(self.input_channels * self.max_len, inner_dim)

        self.fc_mu = nn.Linear(inner_dim, latent_dim)
        self.fc_logvar = nn.Sequential(
            nn.Linear(inner_dim, latent_dim),
            nn.Softplus() # Don't know if this is needed or not
        )

        print("Latent sequence length: ", self.max_len)
        print("Latent hidden channels: ", self.input_channels)
        print("Latent inner dim: ", inner_dim)
        print("Encoder: ", self.encoder)
        print("fc_0: ", self.fc_0)
        print("fc_mu: ", self.fc_mu)
        print("fc_logvar: ", self.fc_logvar)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc_0(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, hidden_channels, input_channels, latent_dim, kernel_size, stride, padding, layers, pooling, max_len, pooling_window, embedding, embedding_dim, pool_doublingtime, conv_doublingtime, upsampling_method, inner_dim):
        super(Decoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.max_len = max_len
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.dec_ref = []
        if embedding:
            self.dec_ref.append(nn.Conv1d(embedding_dim, self.input_channels, kernel_size = 1, stride = 1, padding = 0))
            # self.dec_ref.append(nn.ReLU()) # Im not sure if this ReLU should be there or not. The model seems to improve by 1 %-point when it is absent.
            self.input_channels = embedding_dim
        for i in range(layers):
            if i != 0:
                self.dec_ref.append(nn.LeakyReLU())
            self.dec_ref.append(nn.ConvTranspose1d(self.hidden_channels, self.input_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            self.max_len = int(((self.max_len - kernel_size + 2*padding) / stride + 1) ) # Can give weird values if stride doesn't divide the length
            self.input_channels = self.hidden_channels
            if (i+1) % pool_doublingtime == 0:
                if pooling:
                    self.dec_ref.append(nn.Upsample(size = self.max_len, mode=upsampling_method))
                    if i != layers - 1: # Don't double the length at the last layer, so that we can use the self.max_len value in the forward pass
                        self.max_len = int(np.ceil(self.max_len/pooling_window)) # Upsampling the length. Using ceil! You could choose something different.
            if (i+1) % conv_doublingtime == 0:    
                self.hidden_channels *= 2
        self.dec_ref = self.dec_ref[::-1]
        self.decoder = nn.Sequential(
            *self.dec_ref
        )
        self.fc_1 = nn.Linear(latent_dim, inner_dim)
        self.fc_z = nn.Linear(inner_dim, self.input_channels * self.max_len)

        print("fc1: ", self.fc_1)
        print("fc_z: ", self.fc_z)
        print("Decoder: ", self.decoder)

    def forward(self, z):
        z = self.fc_1(z)
        z = self.fc_z(z)
        z = z.view(-1, self.input_channels, self.max_len)
        x_hat = self.decoder(z)
        return x_hat
    
class cVAE(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim, kernel_size, stride, padding, max_len, layers, pooling, pooling_window, embedding, embedding_dim, pool_doublingtime, conv_doublingtime, pooling_method, upsampling_method, inner_dim = 2048):
        super(cVAE, self).__init__()
        # Encoder
        self.encoder = Encoder(input_channels, hidden_channels, latent_dim, kernel_size, stride, padding, layers, pooling, max_len, pooling_window, embedding, embedding_dim, pool_doublingtime, conv_doublingtime, pooling_method, inner_dim)
        # Decoder
        self.decoder = Decoder(hidden_channels, input_channels, latent_dim, kernel_size, stride, padding, layers, pooling, max_len, pooling_window, embedding, embedding_dim, pool_doublingtime, conv_doublingtime, upsampling_method, inner_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar