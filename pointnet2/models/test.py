import torch
import torch.nn as nn

import etw_pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import PointnetSAModuleMSG, PointnetSAModule

# from pointnet2.models import Pointnet2ClsMSG

class TestNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, use_xyz=True):
        super(TestNet, self).__init__()

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
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.FC_layer(features.squeeze(-1))

# model = Pointnet2ClsMSG(input_channels=0, num_classes=40, use_xyz=True)
model = TestNet(input_channels=0, num_classes=40, use_xyz=True)
model.cuda()

import numpy as np
data = np.random.normal(size=(2,2048,3)).astype(np.float32)
data = torch.tensor(data)

data = data.cuda()
out = model(data)
print(out.shape)
