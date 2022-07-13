import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        # No batchnorm
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=2, out_channels=self.num_features, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=self.num_features, out_channels=self.num_features*2, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(num_features=self.num_features*2),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

        self.enc3 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=self.num_features*2, out_channels=self.num_features*4, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(num_features=self.num_features*4),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

        # No padding and stride=1
        self.enc4 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=self.num_features*4, out_channels=self.num_features*8, kernel_size=4, stride=1, padding=0),
            torch.nn.BatchNorm3d(num_features=self.num_features*8),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

        # TODO: 2 Bottleneck layers

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.num_features*8, out_features=self.num_features*8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.num_features*8, out_features=self.num_features*8),
            torch.nn.ReLU()
        )

        # TODO: 4 Decoder layers

        self.dec1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=self.num_features*8*2, out_channels=self.num_features*4, kernel_size=4, stride=1, padding=0),
            torch.nn.BatchNorm3d(num_features=self.num_features*4),
            torch.nn.ReLU()
        )

        self.dec2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=self.num_features*4*2, out_channels=self.num_features*2, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(num_features=self.num_features*2),
            torch.nn.ReLU()
        )

        self.dec3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=self.num_features*2*2, out_channels=self.num_features, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(num_features=self.num_features),
            torch.nn.ReLU()
        )

        self.dec4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=self.num_features*2 , out_channels=1, kernel_size=4, stride=2, padding=1),
        )


    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        x_e1 = self.enc1(x)
        x_e2 = self.enc2(x_e1)
        x_e3 = self.enc3(x_e2)
        x_e4 = self.enc4(x_e3)

        # Reshape and apply bottleneck layers
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        # Decode
        # TODO: Pass x through the decoder, applying the skip connections in the process
        #print("x:", x.shape)
        #print("x_e4:", x_e4.shape)
        x = self.dec1(torch.cat((x,x_e4), 1))
        #print("x:", x.shape)
        #print("x_e3:", x_e3.shape)
        #print("Concat shape: ", torch.cat((x,x_e3), 1).shape)
        x = self.dec2(torch.cat((x,x_e3), 1))
        #print("x: ", x.shape)
        #print("x_e2: ", x_e2.shape)
        x = self.dec3(torch.cat((x,x_e2), 1))
        #print("x: ", x.shape)
        #print("x_e1: ", x_e1.shape)
        x = self.dec4(torch.cat((x,x_e1), 1))
        #print("X shape: ", x.shape)
        x = torch.squeeze(x, dim=1)
        # TODO: Log scaling
        x = torch.log1p(torch.abs(x))
        return x
