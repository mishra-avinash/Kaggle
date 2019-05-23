import cv2
import numpy as np

from .utils import make_mask_img

train_img_dir = "./data/train/"
test_img_dir = "./data/test/"


def train_generator(df, batch_size, category_num,WIDTH=512, HEIGHT=512):
    img_ind_num = df.groupby("ImageId")["ClassId"].count()
    index = df.index.values[0]
    trn_images = []
    seg_images = []
    for i, (img_name, ind_num) in enumerate(img_ind_num.items()):
        img = cv2.imread(train_img_dir + img_name)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        segment_df = (df.loc[index:index + ind_num - 1, :]).reset_index(drop=True)
        index += ind_num
        if segment_df["ImageId"].nunique() != 1:
            raise Exception("Index Range Error")
        seg_img = make_mask_img(segment_df, category_num,WIDTH, HEIGHT)

        # HWC -> CHW
        img = img.transpose((2, 0, 1))
        # seg_img = seg_img.transpose((2, 0, 1))

        trn_images.append(img)
        seg_images.append(seg_img)
        if ((i + 1) % batch_size == 0):
            yield np.array(trn_images, dtype=np.float32) / 255, np.array(seg_images, dtype=np.int32)
            trn_images = []
            seg_images = []
    if (len(trn_images) != 0):
        yield np.array(trn_images, dtype=np.float32) / 255, np.array(seg_images, dtype=np.int32)


def test_generator(df, WIDTH=512, HEIGHT=512):
    img_names = df["ImageId"].values
    for img_name in img_names:
        img = cv2.imread(test_img_dir + img_name)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        # HWC -> CHW
        img = img.transpose((2, 0, 1))
        yield img_name, np.asarray([img], dtype=np.float32) / 255
