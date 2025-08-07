import os
import sys
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# Model Definitions
# ---------------------------------------

class TransformerEncoder(nn.Module):
    """
    Simple Transformer Encoder:
    - Projects input features into an embedding space
    - Adds a learnable positional encoding
    - Applies stacked TransformerEncoder layers
    - Projects back to the desired output dimension
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        # 1. Linear embedding layer: from input_dim -> embed_dim
        self.embedding = nn.Linear(input_dim, embed_dim)
        # 2. Learnable positional encoding (shape: [1, 1, embed_dim])
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 3. Single TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # input shape will be (batch, seq_len, embed_dim)
        )
        # 4. Stack multiple layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 5. Final projection: embed_dim -> output_dim
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, output_dim)
        """
        # 1. Embed inputs and add positional encoding (broadcast over seq_len)
        x = self.embedding(x) + self.positional_encoding
        # 2. Pass through transformer encoder
        x = self.transformer(x)
        # 3. Project back to output dimension
        return self.fc_out(x)


class MS_Loss(nn.Module):
    """
    Multi-Stage Loss / Neural Operator Module:
    - r attention heads per block
    - nb sequential attention blocks
    - Each block: Q/K projections, score computation, aggregation
    - LayerNorm applied between blocks
    - Final output head per last block
    """
    def __init__(self,
                 tokend: int,
                 r: int,
                 dk: int,
                 nb: int,
                 featured: int,
                 out_dim: int = 1):
        super().__init__()
        self.tokend = tokend          # number of tokens (sequence length)
        self.r = r                    # number of attention heads
        self.dk = dk                  # dimension of Q/K projections
        self.nb = nb                  # number of blocks
        self.featured = featured      # feature dimension
        self.out_dim = out_dim        # final output dimension

        # Create Q/K projection layers and final output heads
        for block in range(nb):
            for head in range(r):
                self.add_module(f"fcq_{block}_{head}", nn.Linear(featured, dk))
                self.add_module(f"fck_{block}_{head}", nn.Linear(featured, dk))
            # Final projection for this block output
            self.add_module(f"fcp_{block}", nn.Linear(tokend, out_dim))
        # LayerNorm modules between blocks (including before first)
        for i in range(nb + 1):
            self.add_module(f"norm_{i}", nn.LayerNorm([tokend, featured]))

    def forward(self, x: torch.Tensor, P: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, tokend, featured)
            P: Optional tensor (unused here)
        Returns:
            out: Tensor of shape (batch_size, tokend, out_dim)
        """
        # Softmax for attention scores
        attn_fn = nn.Softmax(dim=-1)
        # Initial normalization
        h = getattr(self, "norm_0")(x)  # apply LayerNorm
        # Sequential blocks
        for block in range(self.nb):
            # Accumulate multi-head results
            mid = torch.zeros_like(h)
            for head in range(self.r):
                Q = getattr(self, f"fcq_{block}_{head}")(h)  # (batch, tokend, dk)
                K = getattr(self, f"fck_{block}_{head}")(h)  # (batch, tokend, dk)
                # Compute attention scores: (batch, tokend, tokend)
                scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.dk)
                A = attn_fn(scores)
                # Weighted sum
                mid += torch.matmul(A, h)
            # Residual connection + normalization for next block
            h = getattr(self, f"norm_{block+1}")(mid) + h
        # Final output summation over heads' output heads
        out = 0
        for head in range(self.r):
            out = out + getattr(self, f"fcp_{self.nb-1}")(h)  # (batch, tokend, out_dim)
        return out


class CombinedModel(nn.Module):
    """
    Integrates TransformerEncoder + MS_Loss:
    1. Transformer encodes input sequence
    2. Concatenate with initial y sequence
    3. Pass through MS_Loss
    """
    def __init__(self,
                 transformer: TransformerEncoder,
                 nao: MS_Loss):
        super().__init__()
        self.transformer = transformer
        self.nao = nao

    def forward(self, x: torch.Tensor, y_init: torch.Tensor) -> torch.Tensor:
        # 1) Transformer encode x: (batch, seq_len, output_dim)
        z = self.transformer(x)
        # 2) Concatenate original y_init + transformer output
        z_cat = torch.cat([y_init, z], dim=1).double()
        # 3) Neural operator forward
        return self.nao(z_cat)


# ---------------------------------------
# Learning Rate Schedulers
# ---------------------------------------

def linear_scheduler(optimizer: torch.optim.Optimizer, lr: float) -> torch.optim.Optimizer:
    """
    Set optimizer's learning rate to `lr` for all parameter groups.
    """
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return optimizer


def step_lr(initial_lr: float, step: int, step_size: int, gamma: float) -> float:
    """
    Decay learning rate by `gamma` every `step_size` steps.
    """
    return initial_lr * (gamma ** (step // step_size))


# ---------------------------------------
# Logger for Training Curves
# ---------------------------------------

class LearningCurveLogger:
    """
    Logs and saves training & validation loss per epoch.
    """
    def __init__(self):
        self.train_losses = []
        self.test_losses = []

    def update(self, train_loss: float, test_loss: float):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)

    def save(self,
             plot_path: str = "learning_curve.png",
             csv_path: str = "learning_curve.csv"):
        # Plot
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.test_losses,  label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(plot_path)
        plt.close()
        # Save CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "test_loss"])
            for i, (t, v) in enumerate(zip(self.train_losses, self.test_losses)):
                writer.writerow([i+1, t, v])


# ---------------------------------------
# Train/Eval Loops
# ---------------------------------------

def train_loop(model: nn.Module,
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: nn.Module,
               device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch, y_init in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_init  = y_init.to(device)

        optimizer.zero_grad()
        preds = model(x_batch, y_init)
        loss  = criterion(preds.float(), y_batch.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_loop(model: nn.Module,
              dataloader: DataLoader,
              criterion: nn.Module,
              device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch, y_init in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_init  = y_init.to(device)

            preds = model(x_batch, y_init)
            total_loss += criterion(preds.float(), y_batch.float()).item()
    return total_loss / len(dataloader)


# ---------------------------------------
# Main Training Script
# ---------------------------------------

def main():
    # Device and hyperparameters
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs     = 50
    batch_size = 64
    lr         = 1e-3
    step_size  = 10
    gamma      = 0.5

    # Model dimensions
    input_dim  = 15
    output_dim = 3
    embed_dim  = 64
    num_heads  = 4
    num_layers = 2
    r          = 1
    dk         = 32
    nb         = 2
    featured   = input_dim + output_dim

    # ----------------------------------------------------------------
    # Placeholder data: replace with actual loading logic (e.g. MatReader)
    # x_data: (N, seq_len, input_dim)
    # y_data: (N, seq_len, output_dim)
    # y_init: (N, tokend, featured)
    x_data = torch.randn(1000, input_dim, input_dim)
    y_data = torch.randn(1000, output_dim, output_dim)
    y_init = torch.randn(1000, input_dim + output_dim, featured)
    # ----------------------------------------------------------------

    dataset      = TensorDataset(x_data, y_data, y_init)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset, batch_size=batch_size)

    # Instantiate models
    transformer = TransformerEncoder(input_dim, output_dim, embed_dim,
                                     num_heads, num_layers).to(device)
    nao         = MS_Loss(input_dim+output_dim, r, dk, nb, featured,
                           out_dim=output_dim).to(device)
    model       = CombinedModel(transformer, nao).to(device)

    # Optimizer, loss, logger
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    logger    = LearningCurveLogger()

    # Training loop
    for epoch in range(1, epochs+1):
        # Update LR
        current_lr = step_lr(lr, epoch, step_size, gamma)
        linear_scheduler(optimizer, current_lr)

        train_loss = train_loop(model, train_loader, optimizer, criterion, device)
        test_loss  = eval_loop(model, test_loader, criterion, device)

        logger.update(train_loss, test_loss)
        print(f"Epoch {epoch}/{epochs}  LR={current_lr:.5f}  Train Loss={train_loss:.4f}  Test Loss={test_loss:.4f}")

    # Save model & learning curve
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model.pt')
    logger.save()


if __name__ == '__main__':
    main()
