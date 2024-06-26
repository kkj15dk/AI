from torch.utils.data import IterableDataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.functional as F
from sklearn.model_selection import train_test_split
from DNAconversion import *
from MLbuilding import *
from MLvisualisation import *
from MLtesting import *
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) # To remove a warning from logomaker in MLvisualisation.py
import argparse

# Define the parser
parser = argparse.ArgumentParser(description='Train the cVAE')

# Declare arguments
parser.add_argument('--test', type=bool, required=False, default=False)
# parser.add_argument('--test', type=bool, required=False, default=True)
parser.add_argument('--job_id', type=str, required=False, default='test_inner_dim')
parser.add_argument('--models_path', type=str, required=False, default='Models')
parser.add_argument('--plots_path', type=str, required=False, default='Plots')
parser.add_argument('--existing_parameters', required=False, default=None)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--input_channels', type=int, default=21)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--kernel_size', type=int, default=11)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--padding', type=int, default=5)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--pooling', type=bool, default=True)
parser.add_argument('--pooling_window', type=int, default=3)
parser.add_argument('--pooling_method', type=str, choices=['max', 'avg'], default='max')
parser.add_argument('--upsampling_method', type=str, choices=['nearest', 'linear'], default='nearest')
parser.add_argument('--embedding', type=bool, default=True)
parser.add_argument('--embedding_dim', type=int, default=20)
parser.add_argument('--pool_doublingtime', type=int, default=1)
parser.add_argument('--conv_doublingtime', type=int, default=1)
parser.add_argument('--activation_function', type=str, choices=['relu', 'leakyrelu', 'tanh', 'sigmoid', 'gelu'], default='leakyrelu')
parser.add_argument('--inner_dim', type=int, default=None)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--scheduler_step', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--early_stopping_patience', type=int, default=5)
parser.add_argument('--gap_weight', type=float, default=1)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--n_seqs', type=int, default=10000)
parser.add_argument('--random_seed', type=int, default=42)
# parser.add_argument('--aa_file', type=str, default="new4_PKSs.fa")
parser.add_argument('--aa_file', type=str, default="clustalo_alignment.aln")

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
layers = args.layers
pooling = args.pooling
pooling_window = args.pooling_window
embedding = args.embedding
embedding_dim = args.embedding_dim
pool_doublingtime = args.pool_doublingtime
conv_doublingtime = args.conv_doublingtime
pooling_method = args.pooling_method
upsampling_method = args.upsampling_method
activation_function = args.activation_function # Not implemented yet
inner_dim = args.inner_dim

lr = args.lr
scheduler_step = args.scheduler_step
gamma = args.gamma
early_stopping_patience = args.early_stopping_patience
gap_weight = args.gap_weight
num_epochs = args.num_epochs
n_seqs = args.n_seqs

# Set the random seed and file
aa_file = args.aa_file
random_seed = args.random_seed
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
    # seqs = random_aa_seq_unaligned(n_seqs)
    print(seqs[0])
    aa_OHE = one_hot_encode(seqs[0], True)
    print(hot_one_encode(aa_OHE, True))

max_len = max(len(seq) for seq in seqs)
min_len = min(len(seq) for seq in seqs)
print("Number of sequences in dataset:", len(seqs))
print("Longest sequence in dataset:", max_len)
print("Shortest sequence in dataset:", min_len)

if min_len != max_len:
    print("Not all sequences are the same length. Sequences will be padded to the length of the longest sequence. Start '>' and end '<' padding will be used.")
    max_len += 2
    for i, seq in enumerate(seqs):
        seq = '>' + seq + '<'
        seqs[i] = pad_string(seq, max_len, "-") # + 2 for the start and end padding
    print(seqs[0])

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
            aa_condition = inputs_argmax != 20
            gap_condition = inputs_argmax == 20

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
model = cVAE(input_channels,
                hidden_channels,
                latent_dim,
                kernel_size,
                stride,
                padding,
                max_len,
                layers,
                pooling,
                pooling_window,
                embedding,
                embedding_dim,
                pool_doublingtime, 
                conv_doublingtime,
                pooling_method,
                upsampling_method,
                inner_dim=inner_dim
                ).to(DEVICE)
if START_FROM_EXISTING:
    model.load_state_dict(torch.load(f"{args.models_path}/{args.existing_parameters}.pth"))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model parameters: ", count_parameters(model))

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

    cnn_data_label = (train_losses, val_losses, "cVAE")
    val_acc_plotdata = (val_accs, "cVAE")
    val_aa_acc_plotdata = (val_aa_accs, "cVAE")
    val_gap_acc_plotdata = (val_gap_accs, "cVAE")
    quick_loss_plot([cnn_data_label], args.plots_path + "/" + args.job_id + "_loss", "CE + KLD")
    quick_acc_plot([val_acc_plotdata], args.plots_path + "/" + args.job_id + "_val_acc")
    quick_acc_plot([val_aa_acc_plotdata], args.plots_path + "/" + args.job_id + "_val_acc_aa")
    quick_acc_plot([val_gap_acc_plotdata], args.plots_path + "/" + args.job_id + "_val_acc_gap")

    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        best_val_acc = val_avg_acc
        # torch.save(model.state_dict(), best_model_path)
        torch.save(model, best_model_path)
        epoch_since_improvement = 0
    else:
        epoch_since_improvement += 1
        if epoch_since_improvement >= early_stopping_patience:
            print(f"Early stopping. No improvement in {early_stopping_patience} epochs.")
            break  # This will exit the training loop
    
    # Print the average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | Val Acc: {val_avg_acc:.4f} | Val aa Acc: {val_aa_acc:.4f} | Val gap Acc: {val_gap_acc:.4f}| LR: {learning_rate:g}")

# Test the best model
test_model(best_model_path, seqs, latent_dim, DEVICE, samples = 5)