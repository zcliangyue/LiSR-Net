import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from components.edsr_lidar import EDSRLiDAR # encoder
from components.tf_weights import WeightTransformer # decoder
from einops import rearrange

# from models.model_utils import register_model


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers.
    :param shape: shape of image
    :param ranges:
    :param flatten: ture if flatten the coordinates
    """

    coord_seqs = []

    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


# @register_model('iln')

class ILN(nn.Module):
    def __init__(self, d=1, h=8):
        super().__init__()
        self.d = d  # depth
        self.h = h  # num of heads

        # Encoder: EDSR
        self.encoder = EDSRLiDAR(n_resblocks=16, n_feats=64, res_scale=1.0)

        # Attention: ViT
        dim = self.encoder.out_dim
        self.attention = WeightTransformer(num_classes=1, dim=dim,
                                           depth=self.d, heads=self.h, mlp_dim=dim,
                                           dim_head=(dim//self.h), dropout=0.1)

    def gen_feat(self, inp):
        self.inp_img = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_detection(self, coord):
        feat = self.feat
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 2, *feat.shape[-2:])

        # N: batch size
        # Q: query size
        # D: feature dimension
        # T: neighbors
        preds = torch.empty((4, coord.shape[0], coord.shape[1]), device='cuda')
        rel_coords = torch.empty((4, coord.shape[0], coord.shape[1], coord.shape[2]), device='cuda')
        q_feats = torch.empty((4, feat.shape[0], coord.shape[1], feat.shape[1]), device='cuda')

        n, q = coord.shape[:2]
        t = 0
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift  # vertical
                coord_[:, :, 1] += vy * ry + eps_shift  # horizontal
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # q_feat: z_t           [N, Q, D]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # q_coord: q_t          [N, Q, 2]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # rel_coord: del q_t    [N, Q, 2]
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # pred: r_t             [N, Q, 1]
                pred = F.grid_sample(self.inp_img, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                preds[t, :, :] = pred.view(n, q)    # [4, N, Q]
                rel_coords[t, :, :, :] = rel_coord  # [4, N, Q, 2]
                q_feats[t, :, :, :] = q_feat        # [4, N, Q, D]
                t = t + 1

        q_feats = rearrange(q_feats, "t n q d -> (n q) t d")
        rel_coords = rearrange(rel_coords, "t n q c -> (n q) t c")

        weights = self.attention(q_feats, rel_coords)               # [N*Q, 4, 1]
        preds = rearrange(preds, "t n q -> (n q) t").unsqueeze(1)   # [N*Q, 1, 4]

        ret = torch.matmul(preds, weights)  # [N*Q, 1]
        ret = ret.view(n, q, -1)            # [N, Q, 1]

        return ret

    def forward(self, inp, coord):
        self.gen_feat(inp)
        return self.query_detection(coord)




# ILN Lightning Module
class ILNLitModule(LightningModule):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        upscaling_factor: int,
        in_channels: int,
    ) -> None:
        
        """Initialize a `UnetLitModule`.
    
        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = ILN()
        
        # loss function
        self.criterion = torch.nn.L1Loss()

        # acc 
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_acc = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_acc = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_acc = MeanMetric()
        
        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: A tensor of images.
        :return: A tensor of images.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.
        """
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(F.mse_loss(preds, targets))
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(F.mse_loss(preds, targets))
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:

        "Lightning hook that is called when a validation epoch ends."

        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
       
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:

        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:

        optimizer = self.hparams.optimizer(self.trainer.model.parameters())
        
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = ILNLitModule(optimizer=torch.optim.Adam, scheduler=None,
                       upscaling_factor=4, in_channels=1)

