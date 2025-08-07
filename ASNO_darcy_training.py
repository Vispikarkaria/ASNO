import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.fft
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

import operator
from functools import reduce
from functools import partial
import os
from timeit import default_timer
from utilities4 import *
import sys
from VAE import Encoder, Decoder, VAE_model
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.fft
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

import operator
from functools import reduce
from functools import partial
import os
from timeit import default_timer
from utilities4 import *
import sys
from VAE import Encoder, Decoder, VAE_model
import pickle

# Define learning rates, gamma (decay factor), and weight decay
lrs = [3e-3]  # List of learning rates to try
gammas = [0.7]  # List of gamma values (learning rate decay) to try
wds = [1e-2]  # List of weight decay values to try

# Model parameters
nb = 2  # Number of layers in the model
dk = 50  # Dimension of the key/query vectors
sall = dk  # Feature dimension (same as dk here)

# Random seed for reproducibility
seed = 0
torch.manual_seed(seed)  # Set seed for torch (CPU)
torch.cuda.manual_seed(seed)  # Set seed for torch (GPU, if available)
np.random.seed(seed)  # Set seed for numpy

# Other hyperparameters
r = 1  # Number of attention heads (usually in attention models)
trial = 1  # Trial ID (could be useful for multiple runs)
step_size = 100  # Step size for learning rate scheduler
epochs = 25  # Number of training epochs
S = 21  # Spatial resolution (e.g., 21x21 grid)
dt = 5e-5  # Time step size
df = dk  # Feature dimension (same as dk here)

# Data configuration
data_dir = '/home/vnk3019/foundational_model_NAO/data_multi_input_V3.mat'  # Path to the data file

# Dataset splitting
ntotal = 100  # Total number of samples
ntrain = 90  # Number of training samples
ntest = ntotal - ntrain  # Number of test samples

# Time series configuration
sample_per_task = 1  # Number of samples per task (in multi-task learning)
n_timesteps = 100  # Total number of time steps
nt = n_timesteps - 2  # Effective time steps after accounting for history
n_f = sample_per_task * nt  # Number of features (usually calculated from task and time steps)
n_randperm = 10  # Number of random permutations (data augmentation)

# Batch sizes
batch_size_train = n_randperm  # Training batch size
batch_size_test = batch_size_train  # Test batch size (same as training batch size)

# Total number of training and test samples
ntrain_total = ntrain * n_randperm
ntest_total = ntest * n_randperm

# Input/output dimensions
input_dim = df  # Input dimension (feature dimension)
out_dim = S ** 2  # Output dimension (e.g., 21x21 grid)

# Sampling configurations
sample_per_task = 1  # Number of samples per task
rands = 100  # Number of random samples to consider (maybe a setting for randomness)

# Splitting indices for training and testing
train_indexes = np.arange(0, 100)  # Indices for training data
test_indexes = np.arange(0, 100)  # Indices for testing data

# Generate lists for training, validation, and testing
train_list = list(train_indexes)[:ntrain * sample_per_task]
valid_list = list(train_indexes)[-ntest * sample_per_task:]  # Validation data (can be adjusted as needed)
test_list = list(train_indexes)[ntrain * sample_per_task:]

# Load the dataset using MatReader
reader = MatReader(data_dir)

# Extract fields from the dataset for training and testing
sol_train = reader.read_field('sol')[train_list, :].view(ntrain, sample_per_task, n_timesteps, S, S)
f_train = reader.read_field('source')[train_list, :].view(ntrain, sample_per_task, n_timesteps, S, S)
y_train = reader.read_field('chi')[train_list, :].view(ntrain, sample_per_task, n_timesteps, S, S)
sol_prev_train = reader.read_field('sol_old')[train_list, :].view(ntrain, sample_per_task, n_timesteps, S, S)

sol_test = reader.read_field('sol')[test_list, :].view(ntest, sample_per_task, n_timesteps, S, S)
f_test = reader.read_field('source')[test_list, :].view(ntest, sample_per_task, n_timesteps, S, S)
y_test = reader.read_field('chi')[test_list, :].view(ntest, sample_per_task, n_timesteps, S, S)
sol_prev_test = reader.read_field('sol_old')[test_list, :].view(ntest, sample_per_task, n_timesteps, S, S)

# Prepare training data by extracting consecutive time steps (history)
sol_train_u0 = sol_prev_train[:, :, :n_timesteps - 5, ...]
sol_train_u1 = sol_prev_train[:, :, 1:n_timesteps - 4, ...]
sol_train_u2 = sol_prev_train[:, :, 2:n_timesteps - 3, ...]
sol_train_u3 = sol_prev_train[:, :, 3:n_timesteps - 2, ...]
sol_train_u4 = sol_prev_train[:, :, 4:n_timesteps - 1, ...]
sol_train_u5 = sol_prev_train[:, :, 5:n_timesteps, ...]
sol_train_y = y_train[:, :, 5:n_timesteps, ...]

# Reshape and permute tensors to match the required input format
sol_train_u0 = sol_train_u0[:ntrain, ...].reshape(ntrain, 95, S ** 2).permute(0, 2, 1).float()
sol_train_u1 = sol_train_u1[:ntrain, ...].reshape(ntrain, 95, S ** 2).permute(0, 2, 1).float()
sol_train_u2 = sol_train_u2[:ntrain, ...].reshape(ntrain, 95, S ** 2).permute(0, 2, 1).float()
sol_train_u3 = sol_train_u3[:ntrain, ...].reshape(ntrain, 95, S ** 2).permute(0, 2, 1).float()
sol_train_u4 = sol_train_u4[:ntrain, ...].reshape(ntrain, 95, S ** 2).permute(0, 2, 1).float()
sol_train_u5 = sol_train_u5[:ntrain, ...].reshape(ntrain, 95, S ** 2).permute(0, 2, 1).float()
sol_train_y = sol_train_y[:ntrain, ...].reshape(ntrain, 95, S ** 2).permute(0, 2, 1).float()

# Initialize lists to store randomly selected data for each time step
u0_df = []
u1_df = []
u2_df = []
u3_df = []
u4_df = []
u5_df = []
y_df = []

# Number of features to randomly select
n_f = n_timesteps - 5

# Randomly select features and append them to respective lists
for _ in range(n_randperm):
    crand = torch.randperm(n_f)[:df]
    u0_df.append(sol_train_u0[..., crand])
    u1_df.append(sol_train_u1[..., crand])
    u2_df.append(sol_train_u2[..., crand])
    u3_df.append(sol_train_u3[..., crand])
    u4_df.append(sol_train_u4[..., crand])
    u5_df.append(sol_train_u5[..., crand])
    y_df.append(sol_train_y[..., crand])


# Concatenate the selected features into the final training datasets
u0_train = torch.cat(u0_df, dim=0)
u1_train = torch.cat(u1_df, dim=0)
u2_train = torch.cat(u2_df, dim=0)
u3_train = torch.cat(u3_df, dim=0)
u4_train = torch.cat(u4_df, dim=0)
u5_train = torch.cat(u5_df, dim=0)
sol_y_train = torch.cat(y_df, dim=0)

# Normalize the training datasets using a Gaussian normalizer
x_normalizer = GaussianNormalizer(sol_prev_train)
f_normalizer = GaussianNormalizer(sol_y_train)

u0_train = x_normalizer.encode(u0_train)
u1_train = x_normalizer.encode(u1_train)
u2_train = x_normalizer.encode(u2_train)
u3_train = x_normalizer.encode(u3_train)
u4_train = x_normalizer.encode(u4_train)
sol_y_train = f_normalizer.encode(sol_y_train)
u5_train = x_normalizer.encode(u5_train)  ## Not encoding u5_train

u_seq1 = torch.cat((u0_train, u1_train, u2_train, u3_train, u4_train), dim=1) # ntrain*n_randperm X 15 X sall

# Define input and output dimensions based on the concatenated sequence
input_dim = u_seq1.size()[1]
output_dim = u5_train.size()[1]

# Prepare testing data by extracting consecutive time steps (history)
sol_test_u0 = sol_prev_test[:, :, :n_timesteps - 5, ...]
sol_test_u1 = sol_prev_test[:, :, 1:n_timesteps - 4, ...]
sol_test_u2 = sol_prev_test[:, :, 2:n_timesteps - 3, ...]
sol_test_u3 = sol_prev_test[:, :, 3:n_timesteps - 2, ...]
sol_test_u4 = sol_prev_test[:, :, 4:n_timesteps - 1, ...]
sol_test_u5 = sol_prev_test[:, :, 5:n_timesteps, ...]
sol_test_y = y_test[:, :, 5:n_timesteps, ...]

# Reshape and permute the test tensors to match the required input format
sol_test_u0 = sol_test_u0[:ntest, ...].reshape(ntest, 95, S ** 2).permute(0, 2, 1).float()
sol_test_u1 = sol_test_u1[:ntest, ...].reshape(ntest, 95, S ** 2).permute(0, 2, 1).float()
sol_test_u2 = sol_test_u2[:ntest, ...].reshape(ntest, 95, S ** 2).permute(0, 2, 1).float()
sol_test_u3 = sol_test_u3[:ntest, ...].reshape(ntest, 95, S ** 2).permute(0, 2, 1).float()
sol_test_u4 = sol_test_u4[:ntest, ...].reshape(ntest, 95, S ** 2).permute(0, 2, 1).float()
sol_test_u5 = sol_test_u5[:ntest, ...].reshape(ntest, 95, S ** 2).permute(0, 2, 1).float()
sol_test_y = sol_test_y[:ntest, ...].reshape(ntest, 95, S ** 2).permute(0, 2, 1).float()

# Initialize lists to store randomly selected data for each time step
u0_df = []
u1_df = []
u2_df = []
u3_df = []
u4_df = []
u5_df = []
y_df = []

for _ in range(n_randperm):
    crand = torch.randperm(n_f)[:sall]
    u0_df.append(sol_test_u0[..., crand])
    u1_df.append(sol_test_u1[..., crand])
    u2_df.append(sol_test_u2[..., crand])
    u3_df.append(sol_test_u3[..., crand])
    u4_df.append(sol_test_u4[..., crand])
    u5_df.append(sol_test_u5[..., crand])
    y_df.append(sol_test_y[..., crand])


# Concatenate the selected features into the final training datasets
sol_test_u0 = torch.cat(u0_df, dim=0) # ntrain*n_randperm X 3 X sall
sol_test_u1 = torch.cat(u1_df, dim=0)
sol_test_u2 = torch.cat(u2_df, dim=0)
sol_test_u3 = torch.cat(u3_df, dim=0)
sol_test_u4 = torch.cat(u4_df, dim=0)
sol_test_u5 = torch.cat(u5_df, dim=0)
sol_test_y = torch.cat(y_df, dim=0)


# Normalize the test tensors using the same normalizer used for training data
#YY: should use the same normalizer for train and test
#x_normalizer = GaussianNormalizer(sol_prev_test)  # Assuming it is trained on your training dataset

u_test_u0 = x_normalizer.encode(sol_test_u0)
u_test_u1 = x_normalizer.encode(sol_test_u1)
u_test_u2 = x_normalizer.encode(sol_test_u2)
u_test_u3 = x_normalizer.encode(sol_test_u3)
u_test_u4 = x_normalizer.encode(sol_test_u4)
y_test = f_normalizer.encode(sol_test_y)
u_test_u5 = x_normalizer.encode(sol_test_u5)

# Concatenate normalized test tensors incrementally to form the test sequence
#u_seq_test = u_test_u0
#for u_test in [u_test_u1, u_test_u2, u_test_u3, u_test_u4]:
#    u_seq_test = torch.cat((u_seq_test, u_test), dim=1) # ntest X 15 X sall
u_seq_test = torch.cat((u_test_u0, u_test_u1, u_test_u2, u_test_u3, u_test_u4), dim=1) # ntest X 15 X sall

# Define input and output dimensions based on the concatenated test sequence
input_dim = u_seq_test.size()[1]
output_dim = sol_test_u5.size()[1]

# Example usage for prediction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

u_seq = u_seq1.permute(0, 2, 1).float()  # Reshape to desired dimensions
u5_train = u5_train.permute(0, 2, 1)  # Ensure u5_train has the correct dimensions

# DataLoader for training data
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u_seq1, u5_train, sol_y_train),
                                           batch_size=batch_size_train, shuffle=True)

# Prepare the test data for model input
u_seq_test = u_seq_test.permute(0, 2, 1).float()  # Reshape test data to match training data dimensions if needed

u_test_u5 = u_test_u5.permute(0, 2, 1)  # Ensure u5_train has the correct dimensions

# Load the test data into a DataLoader for batch processing
test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u_seq_test, u_test_u5, y_test),
                                              batch_size=batch_size_test, shuffle=False)

# Print input/output dimensions and the number of previous time steps used
print("Input dimension:", input_dim, "Output dimension:", output_dim, "Number of previous timesteps:", int(input_dim / output_dim))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
from timeit import default_timer

# Filename for saving the model
model_filename = "Transformer-NAO"

# Transformer Encoder model definition
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        # Linear layer to embed the input data to a higher dimension
        self.embedding = nn.Linear(input_dim, embed_dim)
        # Positional encoding as a learnable parameter
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Assuming a max length of 50
        # Define a single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        # Stack multiple transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final fully connected layer to map to the output dimension
        self.fc = nn.Linear(embed_dim, output_dim)  # Reduce to (output_dim / seq_len)

    def forward(self, x):
        # Apply embedding and add positional encoding
        x = self.embedding(x)# + self.positional_encoding
        # Pass through the transformer encoder
        x = self.transformer_encoder(x)
        # Apply the fully connected layer
        x = self.fc(x)  # Pooling: mean across time steps
        return x

# MS_Loss model definition
class MS_Loss(nn.Module):
    def __init__(self, tokend, r, dk, nb, featured, out_dim=1):
        super(MS_Loss, self).__init__()
        # Initialize model parameters
        self.tokend = tokend
        self.r = r
        self.dk = dk
        self.nb = nb
        self.featured = featured
        self.out_dim = out_dim

        # Define layers in the model using nested loops
        for i in range(self.r):
            for j in range(self.nb):
                self.add_module('fcq_%d_%d' % (j, i), nn.Linear(featured, self.dk))
                self.add_module('fck_%d_%d' % (j, i), nn.Linear(featured, self.dk))
            self.add_module('fcp%d' % i, nn.Linear(self.tokend + self.out_dim, out_dim))

        # Layer normalization
        for j in range(self.nb + 1):
            self.add_module('fcn%d' % j, nn.LayerNorm([self.tokend + self.out_dim, featured]))

    def forward(self, xy):
        # Softmax activation for attention mechanism
        m = nn.Softmax(dim=2)
        batchsize = xy.shape[0]
        Vinput = self._modules['fcn%d' % 0](xy)
        out_ft = torch.zeros((batchsize, self.featured, self.out_dim), device=xy.device)
        for j in range(self.nb - 1):
            mid_ft = torch.zeros((batchsize, self.tokend + self.out_dim, self.featured), device=xy.device)
            for i in range(self.r):
                Q = self._modules['fcq_%d_%d' % (j, i)](Vinput)
                K = self._modules['fck_%d_%d' % (j, i)](Vinput)
                Attn = m(torch.matmul(Q, torch.transpose(K, 1, 2)) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float)))
                V = (torch.matmul(Attn, Vinput))
                mid_ft = mid_ft + V
            Vinput = self._modules['fcn%d' % (j + 1)](mid_ft) + Vinput

        for i in range(self.r):
            Q = self._modules['fcq_%d_%d' % (self.nb - 1, i)](Vinput)
            K = self._modules['fck_%d_%d' % (self.nb - 1, i)](Vinput)
            Attn = m(torch.matmul(Q, torch.transpose(K, 1, 2)) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float)))
            V = (torch.matmul(Attn[:, :self.tokend + self.out_dim, :], xy[:, :, :]))
            V = V.permute(0, 2, 1)
            out_ft += (self._modules['fcp%d' % i](V))

        return out_ft


# Combined model that integrates the TransformerEncoder and MS_Loss
class CombinedModel(nn.Module):
    def __init__(self, transformer, ms_loss, P):
        super(CombinedModel, self).__init__()
        # Initialize the transformer and MS_Loss models
        self.transformer = transformer
        self.ms_loss = ms_loss
        self.P = P

    def forward(self, inputs, y_inputs):
        # Extract the initial part of the input sequence
        u0 = y_inputs.permute(0, 1, 2)
        # Pass the input through the transformer encoder
        xf = self.transformer(inputs.permute(1, 0, 2))
        # Reshape the transformer output
        xf_transformer_input = xf.permute(1, 2, 0)
        # Concatenate the initial sequence and transformer output
        xf_transformer_input = torch.cat((u0, xf_transformer_input), 1)
        # Pass through MS_Loss model
        outputs = self.ms_loss(xf_transformer_input.double(), self.P).permute(0, 2, 1)
        return outputs

# Learning rate scheduler
def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# Learning rate scheduling function
def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))

# Model and training parameter setup
embed_dim = 882  # Embedding dimension for transformer
num_heads = 441  # Number of attention heads
num_layers = 3  # Number of transformer layers

# Input sequence reshaping
d = u0_train.shape[1]  # This is an add-on to the neural operator

# Initialize transformer encoder and MS_Loss models
transformer_encoder = TransformerEncoder(input_dim, output_dim, embed_dim, num_heads, num_layers).to(device).float()
P = torch.zeros((1, output_dim, output_dim)).to(device)

transformer_input_dim = df
transformer_output_dim = S ** 2

model = MS_Loss(d, r, dk, nb, transformer_input_dim, transformer_output_dim).to(device)
#combined_model = CombinedModel(transformer_encoder, model, P).to(device)

# Define training parameters
num_epochs = 10000  # Number of training epochs
learning_rate = 3e-3  # Initial learning rate
wd = 1e-2  # Weight decay
gamma = 0.7  # Learning rate decay factor

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam(list(model.parameters())+list(transformer_encoder.parameters()), lr=learning_rate, weight_decay=wd)  # Optimizer
criterion = nn.MSELoss()  # Loss function
myloss = LpLoss(size_average=False)  # Custom loss function

# Initialize variables to track the best loss and epoch
train_loss_best = train_loss_lowest = test_loss_best = 1e8
best_epoch = 0

# Loop over epochs
for epoch in range(num_epochs):
    # Adjust learning rate
    current_lr = LR_schedule(learning_rate, epoch, step_size, gamma)
    optimizer = scheduler(optimizer, current_lr)
    
    # Print the learning rate for the current epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Learning Rate: {current_lr:.6f}")
    
    transformer_encoder.train()
    model.train()
    epoch_loss = 0
    t1 = default_timer()  # Start timer
    train_l2 = 0

    # Loop over training batches
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, targets, y_inputs = batch
        inputs, targets, y_inputs = inputs.to(device), targets.to(device), y_inputs.to(device)  # Move data to GPU
        this_batch_size = inputs.shape[0]

        optimizer.zero_grad()  # Clear gradients

        # Forward pass through the model
        input0 = inputs.permute(0, 2, 1).to(device)
        input1 = transformer_encoder(input0)
        input2 = torch.cat((y_inputs.to(device), input1.permute(0, 2, 1)), 1)
        outputs = model(input2.double()).permute(0, 2, 1).to(device)
        outputs = x_normalizer.decode(outputs)  # Decode the outputs

        # Compute the loss
        loss = myloss(outputs.reshape(this_batch_size, -1), x_normalizer.decode(targets).reshape(this_batch_size, -1))

        epoch_loss += loss.item()
        loss.backward()  # Backpropagation

        optimizer.step()  # Update model parameters
        train_l2 += loss.item()

    train_l2 /= ntrain_total  # Normalize the training loss

    # Track the lowest training loss
    if train_l2 < train_loss_lowest:
        train_loss_lowest = train_l2

    # Save the model with the best performance on the validation set
    if train_l2 < train_loss_best:
        model.eval()  # Set model to evaluation mode
        test_l2 = 0.0

        # Evaluate on the test set
        with torch.no_grad():
            for batch in test_dataloader:
                inputs, targets, y_inputs = batch
                this_batch_size = inputs.shape[0]

                # Forward pass through the model
                input0 = inputs.to(device)
                input1 = transformer_encoder(input0)
                input2 = torch.cat((y_inputs.to(device), input1.permute(0, 2, 1)), 1)
                outputs = model(input2.double()).permute(0, 2, 1).to(device)
                outputs = x_normalizer.decode(outputs)  # Decode the outputs

                # Compute test loss
                test_l2 += myloss(outputs.reshape(this_batch_size, -1).to(device), x_normalizer.decode(targets).reshape(this_batch_size, -1).to(device)).item()

        test_l2 /= ntest_total  # Normalize the test loss

        # If the current test loss is the best, save the model
        if test_l2 < test_loss_best:
            best_epoch = epoch
            train_loss_best = train_l2
            test_loss_best = test_l2
            torch.save(model.state_dict(), model_filename)  # Save model state

            t2 = default_timer()  # End timer
            print(f'>> depth{nb}, epoch [{(epoch + 1)}], '
                  f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.5f}, test err: {test_l2:.5f}')
        else:
            t2 = default_timer()
            print(f'>> depth{nb}, epoch [{(epoch + 1)}], '
                  f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.5f}, test err: {test_l2:.5f}, (best: [{best_epoch}], '
                  f'{train_loss_best:.5f}/{test_loss_best:.5f})')
    else:
        t2 = default_timer()
        print(f'>> depth{nb}, epoch [{(epoch + 1)}], '
              f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.5f} (best: [{best_epoch}], '
              f'{train_loss_best:.5f}/{test_loss_best:.5f})')


