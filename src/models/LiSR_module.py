from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.regression.mse import MeanSquaredError
import numpy as np
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


###########################################################################################
#################################        model        #####################################
###########################################################################################

# define double conv block
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, kernel_size=(3,3)):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# define down block
class Down(nn.Module):
    """Downscaling with avgpool then dropout"""

    def __init__(self, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate
        self.pool_conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2,2)),
            nn.Dropout2d(p=self.drop_rate)
        )

    def forward(self, x):
        return self.pool_conv(x)

# define up block
class UpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, strides=2, padding=1, outout_padding=1):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides, padding=padding, output_padding=outout_padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.up(x)


class LiSRNet(torch.nn.Module):
    def __init__(self, upscaling_factor, in_channels) -> None:
        super().__init__()

        self.filters = 64
        self.drop_rate = 0.25
        self.in_channels = in_channels
        self.upscaling_factor = upscaling_factor

        self.up_block1 = UpBlock(in_channels=self.in_channels, out_channels=self.filters, strides=(2,1), outout_padding=(1,0))

        # number of transposed conv in head
        up_block_count = int(np.log2(self.upscaling_factor))

        # create a modulelist to save a series of upblock
        self.up_blocks = nn.ModuleList()
        for _ in range(up_block_count - 1):
            self.up_blocks.append(UpBlock(in_channels=self.filters, out_channels=self.filters, strides=(2,1), outout_padding=(1,0)))
        
        self.doubleconv1 = DoubleConv(self.filters,self.filters)
        self.down1 = Down(self.drop_rate)
        self.doubleconv2 = DoubleConv(self.filters, self.filters*2)
        self.down2 = Down(self.drop_rate)
        self.doubleconv3 = DoubleConv(self.filters*2, self.filters*4)
        self.down3 = Down(self.drop_rate)
        self.doubleconv4 = DoubleConv(self.filters*4, self.filters*8)
        self.down4 = Down(self.drop_rate)
        self.doubleconv5 = DoubleConv(self.filters*8, self.filters*16)
        self.dropout1 = nn.Dropout2d(p=self.drop_rate)
        self.up_block5 = UpBlock(self.filters*16, self.filters*8)
        self.doubleconv6 = DoubleConv(self.filters*16, self.filters*8)
        self.dropout2 = nn.Dropout2d(p=self.drop_rate)
        self.up_block6 = UpBlock(self.filters*8, self.filters*4)
        self.doubleconv7 = DoubleConv(self.filters*8, self.filters*4)
        self.dropout3 = nn.Dropout2d(p=self.drop_rate)
        self.up_block7 = UpBlock(self.filters*4, self.filters*2)
        self.doubleconv8 = DoubleConv(self.filters*4, self.filters*2)
        self.dropout4 = nn.Dropout2d(p=self.drop_rate)
        self.up_block8 = UpBlock(self.filters*2, self.filters)
        self.doubleconv9 = DoubleConv(self.filters*2, self.filters)

        self.regression = nn.Conv2d(in_channels=self.filters, out_channels=1, kernel_size=(1,1))
        

    def forward(self, x):

        x0 = self.up_block1(x)

        for up_block in self.up_blocks:
            x0 = up_block(x0)
        
        x1 = self.doubleconv1(x0)

        x2 = self.down1(x1)
        x2 = self.doubleconv2(x2)

        x3 = self.down2(x2)
        x3 = self.doubleconv3(x3)

        x4 = self.down3(x3)
        x4 = self.doubleconv4(x4)

        y4 = self.down4(x4)

        y4 = self.doubleconv5(y4)
        y4 = self.dropout1(y4)
        y4 = self.up_block5(y4)

        y3 = torch.concatenate([x4,y4], dim=1)
        
        y3 = self.doubleconv6(y3)
        y3 = self.dropout2(y3)
        y3 = self.up_block6(y3)
 
        y2 = torch.concatenate([x3,y3], dim=1)
        y2 = self.doubleconv7(y2)
        y2 = self.dropout2(y2)
        y2 = self.up_block7(y2)

        y1 = torch.concatenate([x2,y2], dim=1)
        y1 = self.doubleconv8(y1)
        y1 = self.dropout2(y1)
        y1 = self.up_block8(y1)

        y0 = torch.concatenate([x1,y1], dim=1)
        y0 =self.doubleconv9(y0)

        outputs = self.regression(y0)

        return outputs


class LiSRLitModule(LightningModule):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        upscaling_factor: int,
        in_channels: int,
    ) -> None:
        """Initialize a `LiSRModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = LiSRNet(upscaling_factor=upscaling_factor,
                           in_channels=in_channels)
        
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
    _ = LiSRLitModule(net=None, optimizer=torch.optim.Adam, scheduler=None,
                       upscaling_factor=4, in_channels=1)
