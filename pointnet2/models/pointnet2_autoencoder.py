import torch
import torch.nn as nn


class Pointnet2Encoder(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Classification network

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

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
            .fc(256, bn=True)
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


class PointnetDecoder(nn.Module):
    def __init__(self, in_dim=256, out_dim=2048):
        super(Pointnet2Encoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model = nn.ModuleList()
        self.model.append(
            nn.linear(256, 1024),
            nn.ReLU(),
            nn.linear(1024, 1024),
            nn.ReLU(),
            nn.linear(1024, out_dim*3)
        )

    def forward(self, pts):
    	pts = self.model(pts)
    	pts = pts.view(-1, self.out_dim, 3)
    	return pts


class Pointnet2AE(nn.Module):
	def __init__(self):
		"""
		The encoder takes as input an array of (B, N, 3),
		  and output
		"""
		self.encoder = Pointnet2Encoder(input_channels=3, use_xyz=True)
		self.decoder = PointnetDecoder()

	def forward(self, pts):
		z = self.encoder(pts)
		pts_reconstruct = self.decoder(z)
		return pts_reconstruct