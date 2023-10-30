from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import  DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


"""
define super-resolusion dataset and DataModule for lightning
"""

class LiSRDataset(Dataset):
    
    # define the data path, value of LR and HR
    def __init__(self, data_dir: str, LR_name: str, HR_name: str):
        
        self.data_dir = os.path.join(data_dir, 'RangeImage')

        # init data path
        self.LR_path = os.path.join(self.data_dir, LR_name)
        self.HR_path = os.path.join(self.data_dir, HR_name)

        self.file_names = os.listdir(self.LR_path)
        
    # get the number of images
    def __len__(self):
        return len(self.file_names)
    
    # get LR and HR images
    def __getitem__(self, index):

        file_name = self.file_names[index]

        file_path_LR = os.path.join(self.LR_path, file_name)
        file_path_HR = os.path.join(self.HR_path, file_name)

        # load the .npy file
        image_np_LR = np.expand_dims(np.load(file_path_LR), axis=0)
        image_np_HR = np.expand_dims(np.load(file_path_HR), axis=0)

        # transform into tensor
        # divide 100 for normalization
        image = torch.Tensor(image_np_LR)/100
        label = torch.Tensor(image_np_HR)/100
        #image = image.permute(2,0,1)
        #label = label.permute(2,0,1)

        return image, label
    

class LiSRDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        LR: str = '16',
        HR: str = '64',
        train_val_test_split: Tuple[float, float, float] = [0.7, 0.2, 0.1],
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir 
        self.LR_name = 'RI_' + LR
        self.HR_name = 'RI_' + HR

        self.tran_val_test_split = train_val_test_split

        self.save_hyperparameters(logger=False)

        # data transformations

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size


    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """


    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            # define the dataset
            dataset = LiSRDataset(self.data_dir, self.LR_name, self.HR_name)

            train_len = np.fix(self.tran_val_test_split[0] * dataset.__len__()).astype(int)
            val_len = np.fix(self.tran_val_test_split[1] * dataset.__len__()).astype(int)
            test_len = dataset.__len__() - train_len - val_len

            # split dataset into data_train, data_test, data_val
            splits = random_split(
                dataset=dataset,
                lengths=(train_len, val_len, test_len),
                generator=torch.Generator().manual_seed(42),
            )

            self.data_train = splits[0]
            self.data_val = splits[1]
            self.data_test = splits[2]

    def train_dataloader(self) -> DataLoader[Any]:

        """Create and return the train dataloader."""

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:

        """Create and return the validation dataloader."""

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:

        """Create and return the test dataloader."""

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        
        pass

    def state_dict(self) -> Dict[Any, Any]:
        
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        
        pass


if __name__ == "__main__":
    _ = LiSRDataModule()
