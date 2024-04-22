from torch.utils.data import IterableDataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.functional as F
from MLbuilding import *
from DNAconversion import *
from sklearn.model_selection import train_test_split
from MLvisualisation import *
from aa_test import *
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) # To remove a warning from logomaker in MLvisualisation.py
import argparse

# Define the parser
parser = argparse.ArgumentParser(description='Train the cVAE')

# Declare arguments
parser.add_argument('--test', type=bool, required=False, default=False)
parser.add_argument('--job_id', type=str, required=False, default='test')
parser.add_argument('--models_path', type=str, required=False, default='Models')
parser.add_argument('--plots_path', type=str, required=False, default='Plots')
parser.add_argument('--existing_parameters', required=False, default=None)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--input_channels', type=int, default=22)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--kernel_size', type=int, default=11)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--padding', type=int, default=5)
args = parser.parse_args()
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

best_model_path = f"{args.models_path}/{args.job_id}_parameters.pth"
print("The model parameters will be saved at: " + best_model_path)
if args.existing_parameters == None:
    START_FROM_EXISTING = False
else:
    print("Using " + args.models_path + args.existing_parameters + ".pth as parameter starting point for the model")
    START_FROM_EXISTING = True

# Set the parameters for the cVAE
batch_size = args.batch_size
input_channels = args.input_channels
hidden_channels = args.hidden_channels
latent_dim = args.latent_dim
kernel_size = args.kernel_size
stride = args.stride
padding = args.padding

lr = 0.00001 # Learning rate of 0.01 seems too high, it ruins the model
scheduler_step = 10
gamma = 0.1 # Learning rate decay
early_stopping_patience = 10
gap_weight = 0.9
num_epochs = 100
n_seqs = 10000

# Set the random seed
random_seed = 42
set_seed(random_seed)

# set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device set as {DEVICE}")

CEweights = torch.cat((torch.ones(input_channels - 1), torch.tensor([gap_weight]))).to(DEVICE) # Weights for the cross entropy loss

# Set the number of decimal places to 2
torch.set_printoptions(precision=2)

class MyIterDataset(IterableDataset):
    def __init__(self, generator_function, seqs, len, transform=None, batch_size=batch_size):
        self.generator_function = generator_function
        self.seqs = seqs
        self.transform = transform
        self.batch_size = batch_size
        self.len = len

    def __iter__(self):
        # Create a generator object
        generator = self.generator_function(self.seqs)
        for item in generator:
            if self.transform:
                item = self.transform(item)
            yield item.float()
    
    def __len__(self):
        return self.len

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

def OHEAAgen(seqs):
    # yield from record_gen
    for seq in seqs:
        seq = torch.tensor(one_hot_encode(seq, aa=True))

        yield seq

def CVAEcollate_fn(batch):
    '''
    Given a batch of data, collate the inputs into their own tensors.
    
    Args:
        batch (list): A list containing the input values.
    
    Returns:
        list: A list containing the input and output tensors.
    '''
    # stack the inputs into a single tensor
    inputs = torch.stack(batch)
    return inputs

def loss_fn(outputs, inputs, mu, logvar, weight = CEweights):
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

    labels = inputs.argmax(dim=1)

    CE = F.cross_entropy(outputs, labels, weight = weight, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return CE + KLD

# Load the data
aa_file = "clustalo_alignment.aln"
train_record_aa = [record for record in SeqIO.parse(aa_file, "fasta")]
train_seq_aa = [str(record.seq) for record in train_record_aa]

print("Number of records in test_record_aa:", len(train_record_aa))
print("Number of non redundant sequences in test_record_aa:", len(set(train_seq_aa)))
print("Using this subset of sequences for training")

# Get the unique sequences
seqs = list(set(train_seq_aa))

# comment this out if you want to use random sequences
if args.test:
    seqs = random_aa_seq(n_seqs)
    print(seqs[0])
    aa_OHE = one_hot_encode(seqs[0], True)
    print(hot_one_encode(aa_OHE, True))

max_len = max(len(seq) for seq in seqs)
print("Number of sequences in dataset:", len(seqs))
print("Longest sequence in dataset:", max_len)

# Shuffle the list
np.random.shuffle(seqs)

# Split the list into training, validation, and test sets
train_seqs, test_seqs = train_test_split(seqs, test_size=0.2, random_state=random_seed)
train_seqs, val_seqs = train_test_split(train_seqs, test_size=0.25, random_state=random_seed)  # 0.25 x 0.8 = 0.2

# Create a SparseDataset
train_dataset = MyIterDataset(OHEAAgen, train_seqs, len(train_seqs))
val_dataset = MyIterDataset(OHEAAgen, val_seqs, len(val_seqs))
test_dataset = MyIterDataset(OHEAAgen, test_seqs, len(test_seqs))

# Create a DataLoader
train_dl = DataLoader(train_dataset, collate_fn=CVAEcollate_fn, batch_size=batch_size)
val_dl = DataLoader(val_dataset, collate_fn=CVAEcollate_fn, batch_size=batch_size)
test_dl = DataLoader(test_dataset, collate_fn=CVAEcollate_fn, batch_size=batch_size)

print("Train len:", train_dataset.len)
print("Val len:", val_dataset.len)
print("Test len:", test_dataset.len)

print("Train_dl:", len(train_dl))
print("Val_dl:", len(val_dl))
print("Test_dl:", len(test_dl))

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim, kernel_size, stride, padding, output_len):
        super(Encoder, self).__init__()
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

class Decoder(nn.Module):
    def __init__(self, hidden_channels, input_channels, kernel_size, stride, padding, output_len):
        super(Decoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.output_len = output_len
        # self.fc_z = nn.Linear(latent_dim, hidden_channels * 2 * self.output_len)
        self.fc_z = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * 2 * self.output_len))
            # # nn.ReLU(),
            # nn.Linear(hidden_channels * 2 * self.output_len, hidden_channels * 2 * self.output_len),
            # nn.ReLU())
        self.decoder = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.ConvTranspose1d(hidden_channels*4, hidden_channels*2, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(hidden_channels*2, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(hidden_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Softmax(dim = 1)
        )

    def forward(self, z):
        z = self.fc_z(z)
        z = z.view(-1, self.hidden_channels * 2, self.output_len)
        x_hat = self.decoder(z)
        return x_hat

class ConvolutionalVAE(nn.Module):
    def __init__(self, input_channels, hidden_channels, latent_dim, kernel_size, stride, padding, max_len):
        super(ConvolutionalVAE, self).__init__()

        # Define the output lengths between different layers of the model. hopefully this will make the model easier to manipulate later on
        self.max_len = max_len
        self.output_len = int(((self.max_len - kernel_size + 2*padding) / stride + 1) ) # Can give weird values if stride doesn't divide the length
        
        print("Output length: ", self.output_len)
        # Encoder
        self.encoder = Encoder(input_channels, hidden_channels, latent_dim, kernel_size, stride, padding, self.output_len)
        # Decoder
        self.decoder = Decoder(hidden_channels, input_channels, kernel_size, stride, padding, self.output_len)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

def train_loop(device, train_dl, model, loss_fn, optimizer):
    model.train()
    train_loss = 0
    total_samples = 0
    for inputs in train_dl:
        batch_size = inputs.size(0)
        total_samples += batch_size
        
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs, mu, logvar = model(inputs)
        loss = loss_fn(outputs, inputs, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / total_samples

def val_loop(device, val_dl, model, loss_fn):
    model.eval()
    val_loss = 0
    total_correct = 0
    total_samples = 0
    aa_correct = 0
    aa_total = 0
    gap_correct = 0
    gap_total = 0
    with torch.no_grad():
        for inputs in val_dl:
            batch_size = inputs.size(0)
            total_samples += batch_size

            inputs = inputs.to(device)
            outputs, mu, logvar = model(inputs)
            val_loss += loss_fn(outputs, inputs, mu, logvar).item()
            
            # Calculate argmax of outputs and inputs
            outputs_argmax = outputs.argmax(dim=1)
            inputs_argmax = inputs.argmax(dim=1)

            # Calculate conditions
            correct_preds = outputs_argmax == inputs_argmax
            aa_condition = inputs_argmax != 21
            gap_condition = inputs_argmax == 21

            # Calculate accuracies
            total_correct += correct_preds.float().sum().item()
            aa_correct += (correct_preds & aa_condition).float().sum().item()
            aa_total += aa_condition.float().sum().item()
            gap_correct += (correct_preds & gap_condition).float().sum().item()
            gap_total += gap_condition.float().sum().item()

            avg_val_loss = val_loss / total_samples
            avg_val_acc = total_correct / (total_samples * max_len)

            val_aa_acc = aa_correct / aa_total
            if gap_total == 0:
                val_gap_acc = 1
            else:
                val_gap_acc = gap_correct / gap_total

        return avg_val_loss, avg_val_acc, val_aa_acc, val_gap_acc

# Instantiate the model
model = ConvolutionalVAE(input_channels,
                         hidden_channels,
                         latent_dim,
                         kernel_size,
                         stride,
                         padding,
                         max_len
                         ).to(DEVICE)
if START_FROM_EXISTING:
    model.load_state_dict(torch.load(f"{args.models_path}/{args.existing_parameters}.pth"))

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define your scheduler
scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=gamma)

optimizer.zero_grad()  # initialize gradients

# Initialize variables
train_losses = []
val_losses = []
val_accs = []
val_aa_accs = []
val_gap_accs = []
best_val_loss = float('inf')
epoch_since_improvement = 0

for epoch in range(num_epochs):

    # Training loop
    learning_rate = scheduler.get_last_lr()[0]
    train_avg_loss = train_loop(DEVICE, train_dl, model, loss_fn, optimizer)
    train_losses.append(train_avg_loss)
    scheduler.step()  # step the scheduler

    # Evaluation loop
    val_avg_loss, val_avg_acc, val_aa_acc, val_gap_acc = val_loop(DEVICE, val_dl, model, loss_fn)
    val_accs.append(val_avg_acc)
    val_aa_accs.append(val_aa_acc)
    val_gap_accs.append(val_gap_acc)
    val_losses.append(val_avg_loss)

    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        best_val_acc = val_avg_acc
        torch.save(model.state_dict(), best_model_path)
        epoch_since_improvement = 0
    else:
        epoch_since_improvement += 1
        if epoch_since_improvement >= early_stopping_patience:
            print(f"Early stopping. No improvement in {early_stopping_patience} epochs.")
            break  # This will exit the training loop
    
    # Print the average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | Val Acc: {val_avg_acc:.4f} | Val aa Acc: {val_aa_acc:.4f} | Val gap Acc: {val_gap_acc:.4f}| LR: {learning_rate:g}")


cnn_data_label = (train_losses, val_losses, "cVAE")
val_acc = (val_accs, "cVAE")
val_aa_acc = (val_aa_accs, "cVAE")
val_gap_acc = (val_gap_accs, "cVAE")
quick_loss_plot([cnn_data_label], args.plots_path + "/" + args.job_id + "_loss", "CE + KLD")
quick_acc_plot([val_acc], args.plots_path + "/" + args.job_id + "_val_acc")
quick_acc_plot([val_aa_acc], args.plots_path + "/" + args.job_id + "_val_acc_aa")
quick_acc_plot([val_gap_acc], args.plots_path + "/" + args.job_id + "_val_acc_gap")

# Load the state of the best model
state_dict = torch.load(best_model_path)

# Apply the state to your model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Generate a random latent vector
latent_vector = torch.randn(1, latent_dim).to(DEVICE)  # Assuming latent_dim is the dimension of the latent space

# Pass the latent vector through the decoder
with torch.no_grad():
    reconstructed_output = model.decoder(latent_vector)

# Convert the reconstructed output to the desired format or representation
sample = reconstructed_output.squeeze().cpu().numpy()  # Assuming the output is a tensor

# Convert the output to binary form
sample = sample.argmax(axis=0)
sample = F.one_hot(torch.tensor(sample), num_classes=22).float().numpy().T

def remove_common_gaps(seq1, seq2):
    new_seq1 = []
    new_seq2 = []

    for aa1, aa2 in zip(seq1, seq2):
        if aa1 != '-' or aa2 != '-':
            new_seq1.append(aa1)
            new_seq2.append(aa2)

    return ''.join(new_seq1), ''.join(new_seq2)

# Convert the one-hot encoded sequence to a string
sample_seq = hot_one_encode(sample, True)
aaseq = seqs[random.randint(0, len(seqs) - 1)]

# Reconstruct the random input
aa_OHE = torch.tensor(one_hot_encode(aaseq, True)).float().to(DEVICE)
aa_OHE = pad_to_length(aa_OHE, max_len)

with torch.no_grad():
    recon_aaseq_OHE = model.decoder(model.encoder(aa_OHE.unsqueeze(0))[0])

recon_aaseq = recon_aaseq_OHE.squeeze().cpu().numpy()
# Convert the output to binary form
recon_aaseq = recon_aaseq.argmax(axis=0)
recon_aaseq = F.one_hot(torch.tensor(recon_aaseq), num_classes=22).float().numpy().T
recon_aaseq = hot_one_encode(recon_aaseq, True)

recon_aaseq_wogaps, aaseq_wogaps = remove_common_gaps(recon_aaseq, aaseq)

print('Random sampl: ' + sample_seq)
print('Random input: ' + aaseq)
print('ReconR input: ' + recon_aaseq)
print('Random input: ' + aaseq_wogaps)
print('ReconR input: ' + recon_aaseq_wogaps)