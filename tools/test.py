#!/usr/bin/env python
from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import time
from typing import TYPE_CHECKING

from random_walk_microstructures import DEFAULT_DTYPE, DEFAULT_DEVICE
from random_walk_microstructures import set_seed, load_data, split_indices, get_outputs_processor, CNN, data_transform
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

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
    parser = ArgumentParser(description='Test surrogate model.', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--percent-train', type=float, default=0.9, help='Fraction of samples used for training during surrogate model training.')
    parser.add_argument('--percent-valid', type=float, default=0.05, help='Fraction of samples used for validation during surrogate model training.')
    parser.add_argument('--seed', type=int, default=3019, help='RNG seed used during surrogate model training.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
    parser.add_argument('--dir', type=Path, default=Path('results'), help='Directory containing the saved model.')

    return parser


def validate_cli_arguments(args: Namespace, parser: ArgumentParser) -> None:
    """Validate command-line interface (CLI) arguments.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments.
    parser : argparse.ArgumentParser
        Parser used to raise errors.

    """
    if args.percent_train <= 0 or args.percent_train > 1:
        parser.error(f'--percent-train must be in (0, 1], got {args.percent_train}.')

    if args.percent_valid <= 0 or args.percent_train + args.percent_valid > 1:
        parser.error(f'--percent-valid must be in (0, {1 - args.percent_train}], got {args.percent_valid}.')

    max_seed = 2**32 - 1
    if args.seed < 0 or args.seed > max_seed:
        parser.error(f'--seed must be in [0, {max_seed}], got {args.seed}.')

    if args.batch_size <= 0:
        parser.error(f'--batch-size must be positive, got {args.batch_size}.')


def main() -> None:
    """Test surrogate model from the command line.

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

    train_indices, _, test_indices = split_indices(num_samples=inputs.shape[0],
                                                   percent_train=args.percent_train,
                                                   percent_valid=args.percent_valid)

    outputs_train = outputs[train_indices]
    inputs_test = inputs[test_indices]
    outputs_test = outputs[test_indices]

    outputs_processor = get_outputs_processor()

    outputs_train = outputs_processor.fit(X=outputs_train)
    outputs_test = outputs_processor.transform(X=outputs_test)

    # Datasets
    test_dataset = TensorDataset(torch.from_numpy(inputs_test).to(dtype=torch.float32),
                                 torch.from_numpy(outputs_test).to(dtype=torch.float32))

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    # Model
    model = CNN().to(dtype=DTYPE, device=DEVICE)

    state_dict = torch.load(f=args.dir / 'model.pt', map_location=DEVICE)
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    # Criterion
    criterion = nn.MSELoss()

    #############################
    ########## Testing ##########
    #############################

    print(f'{"-" * 5}Testing start (device: {DEVICE}){"-" * 5}')

    start_time = time.time()

    running_loss = 0
    for batch in test_loader:
        inputs, targets = batch

        inputs, targets = data_transform(inputs=inputs, outputs=targets)

        inputs = inputs.to(device=DEVICE, non_blocking=True)
        targets = targets.to(device=DEVICE, non_blocking=True)

        with torch.set_grad_enabled(mode=False):
            outputs = model(inputs)

        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(test_loader.dataset)

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    milliseconds = int(elapsed_time % 1 * 1000)

    print(f'Test loss: {epoch_loss}')
    print(f'Time: {minutes:02}:{seconds:02}.{milliseconds:03}')
    print(f'{"-" * 5}Testing end{"-" * 5}')


if __name__ == '__main__':
    main()
