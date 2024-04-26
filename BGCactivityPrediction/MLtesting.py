from Bio import SeqIO
import torch
import torch.nn.functional as F
from DNAconversion import one_hot_encode, hot_one_encode, random_aa_seq_unaligned, pad_string
import random
from MLbuilding import cVAE


def remove_common_gaps(seq1, seq2):
    new_seq1 = []
    new_seq2 = []

    for aa1, aa2 in zip(seq1, seq2):
        if aa1 != '-' or aa2 != '-':
            new_seq1.append(aa1)
            new_seq2.append(aa2)

    return ''.join(new_seq1), ''.join(new_seq2)

def test_model(best_model_path, seqs, latent_dim, DEVICE, samples=1):
        
    # Load the state of the best model
    model = torch.load(best_model_path)

    model.to(DEVICE)
    # Set the model to evaluation mode
    model.eval()
    for i in range(samples):
        # Generate a random latent vector
        latent_vector = torch.randn(1, latent_dim).to(DEVICE)  # Assuming latent_dim is the dimension of the latent space

        # Pass the latent vector through the decoder
        with torch.no_grad():
            reconstructed_output = model.decoder(latent_vector)

        # Convert the reconstructed output to the desired format or representation
        sample = reconstructed_output.squeeze().cpu().numpy()  # Assuming the output is a tensor of shape (num_classes, seq_len)
        num_classes = sample.shape[0]

        # Convert the output to binary form
        sample = sample.argmax(axis=0)
        sample = F.one_hot(torch.tensor(sample), num_classes).float().numpy().T

        # Convert the one-hot encoded sequence to a string
        sample_seq = hot_one_encode(sample, True)

        # Get a radnom amino acid sequence from the test set
        aaseq = seqs[random.randint(0, len(seqs) - 1)]

        # Reconstruct the random input
        aa_OHE = torch.tensor(one_hot_encode(aaseq, True)).float().to(DEVICE)

        with torch.no_grad():
            recon_aaseq_OHE = model.decoder(model.encoder(aa_OHE.unsqueeze(0))[0])

        recon_aaseq = recon_aaseq_OHE.squeeze().cpu().numpy()
        # Convert the output to binary form
        recon_aaseq = recon_aaseq.argmax(axis=0)
        recon_aaseq = F.one_hot(torch.tensor(recon_aaseq), num_classes).float().numpy().T
        recon_aaseq = hot_one_encode(recon_aaseq, True)

        recon_aaseq_wogaps, aaseq_wogaps = remove_common_gaps(recon_aaseq, aaseq)

        print('Sample ' + str(i + 1) + ':' )
        print('Random sampl: ' + sample_seq)
        print('Random input: ' + aaseq)
        print('ReconR input: ' + recon_aaseq)
        print('Random input: ' + aaseq_wogaps)
        print('ReconR input: ' + recon_aaseq_wogaps)

if __name__ == "__main__":
    # set the device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device set as {DEVICE}")
    # # Instantiate the model
    # model = cVAE(input_channels=22,
    #             hidden_channels=32,
    #             latent_dim=10,
    #             kernel_size=11,
    #             stride=1,
    #             padding=5,
    #             max_len=41,
    #             layers=2,
    #             pooling=True,
    #             pooling_window=3,
    #             embedding=True,
    #             embedding_dim=10,
    #             ).to(DEVICE)
    latent_dim = 10
    max_len = 41
    aa_file = "new4_PKSs.fa"
    model_path = "Models/test_unaligned_addedRelU_randompad_test_parameters.pth"
    train_record_aa = [record for record in SeqIO.parse(aa_file, "fasta")]
    train_seq_aa = [str(record.seq) for record in train_record_aa]
    # Get the unique sequences
    seqs = list(set(train_seq_aa))
    seqs = random_aa_seq_unaligned(100) # Only for testing

    # Pad the sequences
    for i, seq in enumerate(seqs):
        seqs[i] = pad_string(seq, max_len, "-")
    print(seqs[0])
    test_model(model_path, seqs, latent_dim, DEVICE, samples=10)