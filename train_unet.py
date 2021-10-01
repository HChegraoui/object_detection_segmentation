import os
from argparse import ArgumentParser

import torch
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Adam

from segmentation.UNet import UNet
from segmentation.loader import DatasetLoader

EPSILON = 1e-32


class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        y_input = torch.log(y_input + EPSILON)
        return cross_entropy(y_input, y_target.long().squeeze(1), weight=self.weight,
                             ignore_index=self.ignore_index)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--train_dataset', required=True, type=str)
    parser.add_argument('--val_dataset', type=str)
    parser.add_argument('--checkpoint_path', required=True, type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--out_channels', default=2, type=int)
    parser.add_argument('--depth', default=5, type=int)
    parser.add_argument('--width', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_freq', default=0, type=int)
    parser.add_argument('--save_model', default=0, type=int)
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--use_bbox', type=bool, default=False)
    args = parser.parse_args()
    kwargs = vars(args)
    return kwargs


if __name__ == '__main__':
    arguments = parse_arguments()

    train_dataset = DatasetLoader(arguments['train_dataset'], use_bbox=arguments['use_bbox'])
    val_dataset = DatasetLoader(arguments['val_dataset'], use_bbox=arguments['use_bbox'])
    conv_depths = [int(arguments['width'] * (2 ** k)) for k in range(arguments['depth'])]
    model = UNet(arguments['in_channels'], arguments['out_channels'], conv_depths, use_bbox=arguments['use_bbox'])
    loss = LogNLLLoss()
    optimizer = Adam(model.parameters(), lr=arguments['learning_rate'])
    results_folder = os.path.join(arguments['checkpoint_path'], arguments['model_name'])
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    model.compile(loss, optimizer, results_folder, device=arguments['device'])
    model.fit_dataset(train_dataset, n_epochs=arguments['epochs'], n_batch=arguments['batch_size'], shuffle=True,
                      val_dataset=val_dataset, save_freq=arguments['save_freq'], save_model=arguments['save_model'],
                      verbose=True)
