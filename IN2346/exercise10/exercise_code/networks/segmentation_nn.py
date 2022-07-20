"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################

        self.pre_model = torchvision.models.alexnet(pretrained=True).features
        for pr in self.pre_model.parameters():
            pr.requires_grad = False

        self.ups = torch.nn.Upsample(size=(240,240), mode='nearest')

        self.conv_layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=num_classes, kernel_size=1)
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x_tmp = x
        x = self.pre_model(x)
        x = self.ups(x)
        x = self.conv_layers(x)


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
