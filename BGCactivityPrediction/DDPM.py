import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from DNAconversion import random_aa_seq_unaligned, one_hot_encode, hot_one_encode, pad_string

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 23
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 40,
    timesteps = 1000,
    objective = 'pred_v'
)

seqs = random_aa_seq_unaligned(100)
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
    for i, seq in enumerate(seqs):
        seq = pad_string(seq, max_len, "-") 
        seqs[i] = torch.tensor(one_hot_encode(seq, True))

# Training
training_seqs = torch.stack(seqs).float()
dataset = Dataset1D(training_seqs)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

loss = diffusion(training_seqs)
loss.backward()

# Or using trainer

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)

trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)
