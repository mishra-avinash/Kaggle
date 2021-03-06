from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import DataLoader, Dataset, sampler
from albumentations import HorizontalFlip, Normalize, Compose
import albumentations as albu
from albumentations.torch import ToTensor
import pandas as pd
from utils import make_mask
import cv2


class SteelDataset(Dataset):
    def __init__(self, df, data_folder, phase, mean, std):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(self.phase, self.mean, self.std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images", image_id)
        img = cv2.imread(image_path)

        augmented = self.transforms(image=img, mask=mask)
        ###
        img = augmented['image']
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)


class TestDataset(Dataset):
    '''Dataset for test prediction'''

    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5),  # only horizontal flip as of now
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
                albu.IAAAdditiveGaussianNoise(p=0.2),
                # albu.IAAPerspective(),
                albu.OneOf(
                    [
                        albu.CLAHE(p=1),
                        albu.RandomBrightnessContrast(p=1),
                        albu.RandomGamma(p=1),
                    ],
                    p=0.9,
                ),
                #
                # albu.OneOf(
                #     [
                #         albu.IAASharpen(p=1),
                #         albu.Blur(blur_limit=3, p=1),
                #         albu.MotionBlur(blur_limit=3, p=1),
                #     ],
                #     p=0.9,
                # ),
                #
                # albu.OneOf(
                #     [
                #         albu.RandomBrightnessContrast(p=1),
                #         albu.HueSaturationValue(p=1),
                #     ],
                #     p=0.9,
                # ),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


def provider(
        data_folder,
        df_path,
        phase,
        seed,
        batch_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_workers=4):
    '''Returns dataloader for the model training'''
    df = pd.read_csv(df_path, engine='python')
    # some preprocessing
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=seed)
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, phase, mean, std)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    return dataloader
