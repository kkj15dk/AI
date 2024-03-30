from torch.utils.data import IterableDataset, DataLoader
from scipy.sparse import csr_matrix
from MLbuilding import *
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import os

batch_size = 2
input_channels = 4
hidden_channels = 16
latent_dim = 100
kernel_size = 3
stride = 1
padding = 1

lr = 0.01
n = 1
num_epochs = 2

# Set the random seed
random_seed = 15
set_seed(15)

# Set the maximum length of the DNA sequences
max_len = 10000

# set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device set as {DEVICE}")

class IterableDataset(IterableDataset):
    def __init__(self, generator, transform=None, batch_size=batch_size):
        self.generator = generator
        self.transform = transform
        self.batch_size = batch_size

    def __iter__(self):
        data = []
        for item in self.generator:
            if self.transform:
                item = self.transform(item)
            data.append(item)
            if len(data) == self.batch_size:
                yield data
                data = []
        if data:
            yield data
        
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

def pad_to_length(tensor, length, padding_value=0):
    if tensor.shape[1] < length:
        tensor = F.pad(tensor, (0, length - tensor.shape[1]), value=padding_value)
    else:
        tensor = tensor[:, :length]
    return tensor

def CVAEcollate_fn(batch, length = max_len, padding_value=0):
    '''
    Given a batch of data, collate the inputs into their own tensors.
    
    Args:
        batch (list): A list containing the input values.
    
    Returns:
        list: A list containing the input and output tensors.
    '''
    # stack the inputs into a single tensor
    batch = batch[0] # Someone please fix this!!!
    inputs = [pad_to_length(item, length, padding_value) for item in batch]
    inputs = torch.stack(inputs)
    return inputs

# Loading data from genbank file
gb_folder = "mibig_gbk_3.1/"
gb_files = [gb_folder + f for f in os.listdir(gb_folder) if f.endswith('.gbk')]

# Shuffle the list
np.random.shuffle(gb_files)

# Split the list into training, validation, and test sets
train_files, test_files = train_test_split(gb_files, test_size=0.2, random_state=random_seed)
train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=random_seed)  # 0.25 x 0.8 = 0.2

# Create a generator fro the files
train_record_gen = record_generator(train_files)
val_record_gen = record_generator(val_files)
test_record_gen = record_generator(test_files)

print(train_record_gen)

# Create a generator for the DNA sequences
train_DNA_gen = OHEDNAgen(train_record_gen)
val_DNA_gen = OHEDNAgen(val_record_gen)
test_DNA_gen = OHEDNAgen(test_record_gen)

print(train_DNA_gen)

# Create a SparseDataset
train_dataset = IterableDataset(train_DNA_gen)#, transform=transform)
val_dataset = IterableDataset(val_DNA_gen)#, transform=transform)
test_dataset = IterableDataset(test_DNA_gen)#, transform=transform)

print(train_dataset)

# Create a DataLoader
train_dataloader = DataLoader(train_dataset, collate_fn=CVAEcollate_fn)
val_dataloader = DataLoader(val_dataset, collate_fn=CVAEcollate_fn)
test_dataloader = DataLoader(test_dataset, collate_fn=CVAEcollate_fn)

print("Train len:", train_dataset)
print("Val len:", val_dataset)
print("Test len:", test_dataset)

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
        self.fc1 = nn.Linear(hidden_channels*2 * 7 * 7, latent_dim)
        self.fc2 = nn.Linear(hidden_channels*2 * 7 * 7, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_channels*2, kernel_size=kernel_size, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels*2, hidden_channels, kernel_size=kernel_size, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels, input_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = x.view(-1, hidden_channels*2 * 7 * 7)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        z = self.reparameterize(mu, logvar)
        z = z.view(-1, latent_dim, 1)
        x = self.decoder(z)
        return x, mu, logvar
    

def CVAE_train_loop(n, DEVICE, train_dl, model, loss_fn, optimizer, train_running_loss):
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
    for i, inputs in enumerate(train_dl):
        # Move the inputs and targets to the device
        inputs = inputs.float().to(DEVICE)
        
        # Forward pass
        outputs, mu, logvar = model(inputs)
        
        # Loss: reconstruction loss + KL divergence
        BCE = F.binary_cross_entropy(outputs, inputs, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD

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

    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for inputs, targets in val_dl: # iterate over the validation dataloader
        inputs = inputs.float().to(DEVICE)
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

# Instantiate the model
model = ConvolutionalVAE(input_channels,
                         hidden_channels,
                         latent_dim,
                         kernel_size,
                         stride,
                         padding
                         ).to(DEVICE)

# Define the loss function
loss_function = nn.BCEWithLogitsLoss()

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
    train_avg_loss = CVAE_train_loop(n, DEVICE, train_dataloader, model, loss_function, optimizer, train_running_loss)
    train_losses.append(train_avg_loss)

    # Evaluation loop
    model.eval()  # put model in evaluation mode
    with torch.no_grad():  # disable gradient computation
        val_avg_losses, all_preds, all_labels = CVAE_val_loop(DEVICE, val_dataloader, model, loss_function, val_running_loss, all_preds, all_labels)
        val_losses.append(val_avg_losses)
    # Print the average loss for the epoch
    val_acc = np.mean(np.array(all_preds).flatten() == np.array(all_labels).flatten())
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_losses:.4f} | Val Acc: {val_acc:.4f}")
