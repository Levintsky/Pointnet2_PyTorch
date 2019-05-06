import torch
import torch.nn as nn
import torch.nn.functional as F

import etw_pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import PointnetSAModuleMSG, PointnetSAModule


class Pointnet2Encoder(nn.Module):
    def __init__(self, input_channels=3, use_xyz=True):
        super(Pointnet2Encoder, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[
                    [input_channels, 32, 32, 64],
                    [input_channels, 64, 64, 128],
                    [input_channels, 64, 96, 128],
                ],
                use_xyz=use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(mlp=[128 + 256 + 256, 256, 512, 1024], use_xyz=use_xyz)
        )

        self.FC_layer = (
            pt_utils.Seq(1024)
            .fc(512, bn=True)
            .dropout(0.5)
            .fc(256)
            # .dropout(0.5)
            # .fc(num_classes, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.FC_layer(features.squeeze(-1))


class Pointnet2Decoder(nn.Module):
    def __init__(self, in_dim=256, out_dim=2048):
        super(Pointnet2Decoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(256, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.linear3 = nn.Linear(1024, out_dim * 3)
        """
        self.model = nn.ModuleList()
        self.model.append(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim*3)
        )
        """

    def forward(self, pts):
        # pts = self.model(pts)
        pts = self.linear1(pts)
        pts = F.relu(self.bn1(pts))
        pts = self.linear2(pts)
        pts = F.relu(self.bn2(pts))
        pts = self.linear3(pts)
        pts = pts.view(-1, self.out_dim, 3)
        return pts


class Pointnet2AE(nn.Module):
    def __init__(self):
        """
        The encoder takes as input an array of (B, N, 3),
          and output
        """
        super(Pointnet2AE, self).__init__()
        self.encoder = Pointnet2Encoder(input_channels=0, use_xyz=True)
        self.decoder = Pointnet2Decoder()

    def forward(self, pts):
        z = self.encoder(pts)
        print(z.shape)
        pts_reconstruct = self.decoder(z)
        return pts_reconstruct


if __name__ == "__main__":
    # model = Pointnet2Encoder(input_channels=0, use_xyz=True)
    model = Pointnet2AE()
    model.cuda()

    import numpy as np
    data = np.random.normal(size=(2,2048,3)).astype(np.float32)
    data = torch.tensor(data).cuda()

    out = model(data)
    print(out.shape)
    """
    model = Pointnet2AE().cuda()
    data = np.random.normal(size=(2, 2048, 3)).astype(np.float32)
    data = torch.tensor(data).cuda()
    out = model(data)
    """
