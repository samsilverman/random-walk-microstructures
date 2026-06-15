#!/usr/bin/env python
from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import List, TYPE_CHECKING

from matplotlib import pyplot as plt
from random_walk_microstructures import DEFAULT_DTYPE, DEFAULT_DEVICE
from random_walk_microstructures import set_seed, load_data, split_indices, get_outputs_processor, CNN, print_model, data_transform, load_checkpoint, save_checkpoint
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

if TYPE_CHECKING:
    from argparse import Namespace

DTYPE = DEFAULT_DTYPE
DEVICE = DEFAULT_DEVICE


def build_parser() -> ArgumentParser:
    """Command-line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line interface.

    """
    parser = ArgumentParser(description='Train surrogate model.', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--percent-train', type=float, default=0.9, help='Fraction of samples used for training.')
    parser.add_argument('--percent-valid', type=float, default=0.05, help='Fraction of samples used for validation.')
    parser.add_argument('--seed', type=int, default=3019, help='RNG seed.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--out-dir', type=Path, default=Path('results'), help='Output directory for the saved model.')
    parser.add_argument('--resume', action='store_true', help='Resume training.')

    return parser


def visualize_losses(train_losses: List[float], valid_losses: List[float], file: Path) -> None:
        """Visualize losses from training.

        Parameters
        ----------
        train_losses : List[float]
            Training losses at each epoch.
        valid_losses : List[float]
            Validation losses at each epoch.
        file : pathlib.Path
            File. Must have `.png` extension.

        """
        _, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(6.4, 4.8))

        epochs = range(len(train_losses))

        ax.plot(epochs, train_losses, label='Training loss')
        ax.plot(epochs, valid_losses, label='Validation loss')

        ax.set_title('Loss Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend(loc='best')

        plt.savefig(file)
        plt.show()


def validate_cli_arguments(args: Namespace, parser: ArgumentParser) -> None:
    """Validate command-line interface (CLI) arguments.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments.
    parser : argparse.ArgumentParser
        Parser used to raise errors.

    """
    if args.epochs <= 0:
        parser.error(f'--epochs must be positive, got {args.epochs}.')

    if args.percent_train <= 0 or args.percent_train > 1:
        parser.error(f'--percent-train must be in (0, 1], got {args.percent_train}.')

    if args.percent_valid <= 0 or args.percent_train + args.percent_valid > 1:
        parser.error(f'--percent-valid must be in (0, {1 - args.percent_train}], got {args.percent_valid}.')

    max_seed = 2**32 - 1
    if args.seed < 0 or args.seed > max_seed:
        parser.error(f'--seed must be in [0, {max_seed}], got {args.seed}.')

    if args.batch_size <= 0:
        parser.error(f'--batch-size must be positive, got {args.batch_size}.')

    if args.lr <= 0:
        parser.error(f'--lr must be positive, got {args.lr}.')

    if args.resume:
        args.resume = (args.out_dir / 'checkpoint.pt').is_file()


def main() -> None:
    """Train surrogate model from the command line.

    """
    #############################################
    ########## CLI argument validation ##########
    #############################################

    parser = build_parser()
    args = parser.parse_args()

    validate_cli_arguments(args=args, parser=parser)

    ###########################
    ########## Setup ##########
    ###########################

    set_seed(seed=args.seed)

    # Data preprocessing
    inputs, outputs = load_data()

    train_indices, valid_indices, _ = split_indices(num_samples=inputs.shape[0],
                                                    percent_train=args.percent_train,
                                                    percent_valid=args.percent_valid)

    inputs_train = inputs[train_indices]
    outputs_train = outputs[train_indices]
    inputs_valid = inputs[valid_indices]
    outputs_valid = outputs[valid_indices]

    outputs_processor = get_outputs_processor()

    outputs_train = outputs_processor.fit_transform(X=outputs_train)
    outputs_valid = outputs_processor.transform(X=outputs_valid)

    # Datasets
    train_dataset = TensorDataset(torch.from_numpy(inputs_train).to(dtype=DTYPE),
                                  torch.from_numpy(outputs_train).to(dtype=DTYPE))

    valid_dataset = TensorDataset(torch.from_numpy(inputs_valid).to(dtype=DTYPE),
                                  torch.from_numpy(outputs_valid).to(dtype=DTYPE))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)
    
    # Model
    model = CNN().to(dtype=DTYPE, device=DEVICE)

    print_model(model)

    # Criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    ##############################
    ########## Training ##########
    ##############################

    def run_single_epoch(data_loader: DataLoader, grad_enabled: bool) -> float:
        """Training/validation logic for a single epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader.
        grad_enabled : bool
            Set to `True` for training. Set to `False` for validation.

        Returns
        -------
        loss : float
            Average loss for the epoch.

        """
        if grad_enabled:
            model.train()
        else:
            model.eval()

        running_loss = 0
        for batch in data_loader:
            inputs, targets = batch

            # Apply data transformation (if applicable) to inputs only
            if grad_enabled:
                inputs, targets = data_transform(inputs=inputs, outputs=targets)

            inputs = inputs.to(device=DEVICE, non_blocking=True)
            targets = targets.to(device=DEVICE, non_blocking=True)

            if grad_enabled:
                optimizer.zero_grad()

            with torch.set_grad_enabled(mode=grad_enabled):
                outputs = model(inputs)

            loss = criterion(outputs, targets)

            if grad_enabled:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        return epoch_loss

    best_loss = float('inf')
    best_epoch = 0
    best_state_dict = deepcopy(model.state_dict())

    train_losses = []
    valid_losses = []

    checkpoint_file = args.out_dir / 'checkpoint.pt'
    model_file = args.out_dir / 'model.pt'
    losses_file = args.out_dir / 'losses.png'
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    if args.resume:
        model_state_dict, best_state_dict, optimizer_state_dict, train_losses, valid_losses = load_checkpoint(file=checkpoint_file)

        if len(valid_losses) > 0:
            best_loss = min(valid_losses)
            best_epoch = valid_losses.index(best_loss) + 1

        model.load_state_dict(state_dict=model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    # Training loop
    start_epoch = len(train_losses) + 1
    progress = trange(start_epoch, args.epochs + 1, desc='Training', unit='epoch')
    progress.set_postfix(best_epoch=best_epoch, best_loss=f'{best_loss:.2e}')

    for epoch in progress:
        train_loss = run_single_epoch(data_loader=train_loader, grad_enabled=True)
        valid_loss = run_single_epoch(data_loader=valid_loader, grad_enabled=False)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())

            save_checkpoint(model_state_dict=model.state_dict(),
                            best_model_state_dict=best_state_dict,
                            optimizer_state_dict=optimizer.state_dict(),
                            train_losses=train_losses,
                            valid_losses=valid_losses,
                            file=checkpoint_file)

        progress.set_postfix(best_epoch=best_epoch, best_loss=f'{best_loss:.4e}')

    # Save best model and checkpoint for end of training 
    save_checkpoint(model_state_dict=model.state_dict(),
                    best_model_state_dict=best_state_dict,
                    optimizer_state_dict=optimizer.state_dict(),
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                    file=checkpoint_file)
    torch.save(obj=best_state_dict, f=model_file)

    visualize_losses(train_losses=train_losses, valid_losses=valid_losses, file=losses_file)


if __name__ == '__main__':
    main()
