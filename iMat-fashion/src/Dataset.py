import torch
from . import data_utils


loaders = {
    'pil': data_utils.pil_loader,
    'tns': data_utils.tensor_loader,
    'npy': data_utils.numpy_loader,
    'tif': data_utils.tif_loader,
    'io': data_utils.io_loader
}


class FileDataset(torch.utils.data.Dataset):
    def __init__(self, fpaths,
                 img_loader='pil',
                 targets=None,
                 transform=None,
                 target_transform=None):
        self.fpaths = fpaths
        self.loader = self._get_loader(img_loader)
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def _get_loader(self, loader_type):
        return loaders[loader_type]

    def _get_target(self, index):
        if self.targets is None:
            return torch.FloatTensor(1)
        target = self.targets[index]
        if self.target_transform is not None:
            return self.target_transform(target)
        return torch.FloatTensor(target)

    def _get_input(self, index):
        img_path = self.fpaths[index]
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        input_ = self._get_input(index)
        target = self._get_target(index)
        img_path = self.fpaths[index]
        return input_, target, img_path

    def __len__(self):
        return len(self.fpaths)