from torch.utils.data import IterableDataset, DataLoader
from MLbuilding import *
from sklearn.model_selection import train_test_split
from Bio import SeqIO
from MLvisualisation import *
from aa_test import random_aa_seq
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) # To remove a warning from logomaker in MLvisualisation.py

batch_size = 5
input_channels = 21
hidden_channels = 32
latent_dim = 100
kernel_size = 3
stride = 1
padding = 1

lr = 0.01
num_epochs = 10

# Set the random seed
random_seed = 15
set_seed(15)

# Set the maximum length of the sequences (divisible by 4 for easy twice halving through the CVAE network)
# max_len = 3600
max_len = 520

# set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device set as {DEVICE}")

class IterableDataset(IterableDataset):
    def __init__(self, generator_function, records, len, transform=None, batch_size=batch_size):
        self.generator_function = generator_function
        self.records = records
        self.transform = transform
        self.batch_size = batch_size
        self.len = len


    def __iter__(self):
        data = []
        # Create a new record_generator for each epoch
        record_generator = self.records
        # Pass the new record_generator to the generator function
        generator = self.generator_function(record_generator)
        for item in generator:
            if self.transform:
                item = self.transform(item)
            data.append(item)
            if len(data) == self.batch_size:
                yield data
                data = []
        if data:
            yield data
    
    def __len__(self):
        return self.len
        
def record_generator(files):
    # This function will yield genbank records from a list of files
    for file in files:
        for record in SeqIO.parse(file, "genbank"):
            yield record

def OHEDNAgen(record_gen):
    # yield from record_gen
    for record in record_gen:
        seq = torch.tensor(one_hot_encode(record.seq))
        yield seq

def OHEAAgen(record_gen):
    # yield from record_gen
    for record in record_gen:
        # seq = torch.tensor(one_hot_encode(record.seq, aa=True))
        seq = torch.tensor(one_hot_encode(record, aa=True))
        # # Find the indices of non-zero elements
        # indices = torch.nonzero(seq).t()

        # # Get the values of the non-zero elements
        # values = seq[indices[0], indices[1]]

        # # Create a sparse tensor
        # seq = torch.sparse_coo_tensor(indices, values, seq.size())

        yield seq

def pad_to_length(tensor, length, padding_value=0):
    if tensor.shape[1] < length:
        tensor = F.pad(tensor, (0, length - tensor.shape[1]), value=padding_value)
    else:
        tensor = tensor[:, :length]
    return tensor
    
def loss_fn(outputs, inputs, mu, logvar):
    """
    Calculates the loss function for a Conditional Variational Autoencoder (CVAE).

    Args:
        outputs (torch.Tensor): The predicted outputs from the CVAE.
        inputs (torch.Tensor): The input data to the CVAE.
        mu (torch.Tensor): The mean of the latent space distribution.
        logvar (torch.Tensor): The logarithm of the variance of the latent space distribution.

    Returns:
        torch.Tensor: The calculated loss value.

    """
    BCE = F.binary_cross_entropy_with_logits(outputs, inputs, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def CVAEcollate_fn(batch, length = max_len, padding_value=0):
    '''
    Given a batch of data, collate the inputs into their own tensors.
    
    Args:
        batch (list): A list containing the input values.
    
    Returns:
        list: A list containing the input and output tensors.
    '''
    # stack the inputs into a single tensor
    batch = batch[0] # Someone please fix this!!! Why is batch a list with one element?
    inputs = [pad_to_length(item, length, padding_value) for item in batch]
    inputs = torch.stack(inputs)
    return inputs

# aa_file = "Ascomycete_T1-PKSs.fa"
# train_record_aa = [record for record in SeqIO.parse(aa_file, "fasta")]
# train_record_aa = [record for record in train_record_aa if len(record.seq) <= max_len] # filter out sequences longer than max_len

aa_file = random_aa_seq(1000)
train_record_aa = aa_file


print("Number of records in test_record_aa:", len(train_record_aa))
print("Longest sequence in test_record_aa:", max(len(record) for record in train_record_aa))
# print("Longest sequence in test_record_aa:", max(len(record.seq) for record in train_record_aa))


def plot_train_test_hist(train_record_aa,bins=20):
    ''' Check distribution of train/test scores, sanity check that its not skewed'''
    list = [len(record) for record in train_record_aa]
    # list = [len(record.seq) for record in train_record_aa]
    plt.hist(list, bins=bins, label='train', alpha=1)
    plt.legend()
    plt.xlabel("seq length",fontsize=14)
    plt.ylabel("count",fontsize=14)
    plt.savefig("train_hist.png")

plot_train_test_hist(train_record_aa)

# weird = []

# for record in test_record_aa:
#     allowed = set("ACDEFGHIKLMNPQRSTVWY*XBJ")
#     if not set(record.seq).issubset(allowed):
#         weird.append(record)
#         print(set(record.seq) - allowed)

# print(weird[0].seq)

# Shuffle the list
np.random.shuffle(train_record_aa)

# Split the list into training, validation, and test sets
train_files, test_files = train_test_split(train_record_aa, test_size=0.2, random_state=random_seed)
train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=random_seed)  # 0.25 x 0.8 = 0.2

# Create a SparseDataset
train_dataset = IterableDataset(OHEAAgen, train_files, len(train_files))
val_dataset = IterableDataset(OHEAAgen, val_files, len(val_files))
test_dataset = IterableDataset(OHEAAgen, test_files, len(test_files))

print("train iterable dataset:", train_dataset)

# Create a DataLoader
train_dataloader = DataLoader(train_dataset, collate_fn=CVAEcollate_fn)
val_dataloader = DataLoader(val_dataset, collate_fn=CVAEcollate_fn)
test_dataloader = DataLoader(test_dataset, collate_fn=CVAEcollate_fn)

print("Train len:", train_dataset.len)
print("Val len:", val_dataset.len)
print("Test len:", test_dataset.len)

class ConvolutionalVAE(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim, kernel_size, stride, padding):
        super(ConvolutionalVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Latent vectors mu and logvar
        self.fc1 = nn.Linear(hidden_channels*2 * max_len//4, latent_dim)
        self.fc2 = nn.Linear(hidden_channels*2 * max_len//4, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_channels*2 * max_len//4)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels*2, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = x.view(-1, hidden_channels*2 * max_len//4)
        # print(x.shape)
        mu = self.fc1(x)
        # print(mu.shape)
        logvar = self.fc2(x)
        # print(logvar.shape)
        z = self.reparameterize(mu, logvar)
        # print(z.shape)
        z = self.fc3(z)
        # print(z.shape)
        z = z.view(-1, hidden_channels*2, max_len//4)
        # print(z.shape)
        x = self.decoder(z)
        # print(x.shape)
        return x, mu, logvar

def CVAE_train_loop(DEVICE, train_dl, model, loss_fn, optimizer, train_running_loss):
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
    model.train()
    train_running_loss = 0.0
    for inputs in train_dl:
        # Move the inputs and targets to the device
        inputs = inputs.float().to(DEVICE)
        
        # Forward pass
        outputs, mu, logvar = model(inputs)
        
        # Loss: reconstruction loss + KL divergence
        loss = loss_fn(outputs, inputs, mu, logvar)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()

    # Calculate the average loss for the epoch
    train_avg_loss = train_running_loss / len(train_dl.dataset)
    return train_avg_loss

def CVAE_val_loop(DEVICE, val_dl, model, loss_fn, val_running_loss, all_preds, all_targets, early_stopping_epochs=5):
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
    model.eval()
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for inputs in val_dl: # iterate over the validation dataloader
        inputs = inputs.float().to(DEVICE)

        # Forward pass
        outputs, mu, logvar = model(inputs)
        loss = loss_fn(outputs, inputs, mu, logvar)

        # Get predictions
        preds = torch.sigmoid(outputs) > 0.5 # threshold the output to get the class prediction
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(inputs.cpu().numpy())

        # Calculate loss
        val_running_loss += loss.item()

    val_avg_loss = val_running_loss / len(val_dl)
    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_since_improvement += 1
        if epochs_since_improvement == early_stopping_epochs:
            print('Early stopping')
            return val_avg_loss, all_preds, all_targets
    return val_avg_loss, all_preds, all_targets

# Instantiate the model
model = ConvolutionalVAE(input_channels,
                         hidden_channels,
                         latent_dim,
                         kernel_size,
                         stride,
                         padding
                         ).to(DEVICE)

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
    train_avg_loss = CVAE_train_loop(DEVICE, train_dataloader, model, loss_fn, optimizer, train_running_loss)
    train_losses.append(train_avg_loss)

    # Evaluation loop
    model.eval()  # put model in evaluation mode
    with torch.no_grad():  # disable gradient computation
        val_avg_losses, all_preds, all_labels = CVAE_val_loop(DEVICE, val_dataloader, model, loss_fn, val_running_loss, all_preds, all_labels)
        val_losses.append(val_avg_losses)
    # Print the average loss for the epoch
    val_acc = np.mean(np.array(all_preds).flatten() == np.array(all_labels).flatten())
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_losses:.4f} | Val Acc: {val_acc:.4f}")

cnn_data_label = (train_losses, val_losses, "CNN")
quick_loss_plot([cnn_data_label])

conv_layers, model_weights, bias_weights = get_conv_layers_from_model(model)
view_filters(model_weights)

# just use some seqs from test_df to activate filters
# seqs = [record.seq for record in test_files]
seqs = [record for record in test_files]
some_seqs = random.choices(seqs, k=2000)

filter_activations = get_filter_activations(some_seqs, conv_layers[0], DEVICE, input_channels=["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "*"], aa=True)
view_filters_and_logos(model_weights,filter_activations, input_channels=["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "*"])

def generate_digit(mean, var):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(DEVICE)
    x_decoded = model.decode(z_sample)
    digit = x_decoded.detach().cpu().reshape(28, 28) # reshape vector to 2d array
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()

generate_digit(0.0, 1.0), generate_digit(1.0, 0.0)