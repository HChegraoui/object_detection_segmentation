import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from .blocks import *

from torchsummary import summary

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depths=(64, 128, 256, 512, 1024), use_bbox = True):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'

        super(UNet, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First2D(in_channels, conv_depths[0], conv_depths[0]))
        encoder_layers.extend([Encoder2D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1])
                               for i in range(len(conv_depths)-2)])

        bbox_layers = []
        bbox_layers.append(First2D(1, conv_depths[0], conv_depths[0]))
        bbox_layers.extend([Encoder2D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1])
                               for i in range(len(conv_depths)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i])
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last2D(2*conv_depths[0], conv_depths[0], out_channels))

        # encoder, center and decoder layers
        self.use_bbox = use_bbox
        self.encoder_layers = nn.Sequential(*encoder_layers)
        if(self.use_bbox):
            self.mask_layers = nn.Sequential(*bbox_layers)
        self.center = Center2D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)



    def forward(self, x, return_all=False, bbox=None):
        x_enc = [x]
        if(self.use_bbox):
            mask_enc = [bbox]
            for mask_layer in self.mask_layers:
                mask_enc.append(mask_layer(mask_enc[-1]))
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            if(self.use_bbox):
                x_opposite = x_opposite * mask_enc[-1 - dec_layer_idx]
            x_cat = torch.cat(
                [self.pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            return x_dec[-1]
        else:
            return x_enc + x_dec

    def pad_to_shape(self, this, shp):
        """
        Pads this image with zeroes to shp.
        Args:
            this: image tensor to pad
            shp: desired output shape

        Returns:
            Zero-padded tensor of shape shp.
        """
        if len(shp) == 4:
            pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
        elif len(shp) == 5:
            pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
        return F.pad(this, pad)

    def compile(self,loss, optimizer, checkpoint_folder: str,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 device: torch.device = torch.device('cpu')):
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_folder = checkpoint_folder
        if not os.path.isdir(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        # moving net and loss to the selected device
        self.device = device
        self.to(device=self.device)
        try:
            self.loss.to(device=self.device)
        except:
            pass

    def fit_epoch(self, dataset, n_batch=1, shuffle=False):

        self.train(True)

        epoch_running_loss = 0

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(
                DataLoader(dataset, batch_size=n_batch, shuffle=shuffle)):
            bbox_batch = Variable(rest[0].to(device=self.device)) if self.use_bbox else None
            X_batch = Variable(X_batch.to(device=self.device))
            y_batch = Variable(y_batch.to(device=self.device))
            # training
            self.optimizer.zero_grad()
            y_out = self(X_batch, bbox=bbox_batch)
            training_loss = self.loss(y_out, y_batch)
            training_loss.backward()
            self.optimizer.step()
            epoch_running_loss += training_loss.item()

        self.train(False)

        del X_batch, y_batch

        logs = {'train_loss': epoch_running_loss / (batch_idx + 1)}

        return logs