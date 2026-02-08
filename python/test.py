#!/usr/bin/env python3
"""Script for CNN testing.

"""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from ml import load_data, get_outputs_processor, CNN, data_transform
from utils import set_seed, split_indices, print_model, Tester, load_state_dict


def main() -> None:
    # Setup
    seed = 3019
    set_seed(seed=seed)

    # Data preprocessing
    inputs, outputs = load_data()

    train_indices, _, test_indices = split_indices(num_samples=inputs.shape[0],
                                                   percent_train=0.9,
                                                   percent_valid=0.05)

    outputs_train = outputs[train_indices]
    inputs_test = inputs[test_indices]
    outputs_test = outputs[test_indices]

    outputs_processor = get_outputs_processor()

    outputs_processor.fit(outputs_train)

    outputs_test = outputs_processor.transform(outputs_test)

    # Dataset
    test_dataset = TensorDataset(torch.from_numpy(inputs_test).to(dtype=torch.float32),
                                 torch.from_numpy(outputs_test).to(dtype=torch.float32))

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=64,
                             shuffle=True)

    # Model
    model = CNN()

    model_dir = Path(__file__).resolve().parent.parent / 'models'
    load_state_dict(model=model, file=model_dir / 'model.pt')

    print_model(model)

    # Criterion
    criterion = nn.MSELoss()

    # Test
    tester = Tester(model=model,
                    criterion=criterion,
                    test_loader=test_loader,
                    data_transform=data_transform)

    tester.test(verbose=True)


if __name__ == '__main__':
    main()
