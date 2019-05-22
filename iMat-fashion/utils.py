import numpy as np
import cv2
from itertools import groupby

def make_onehot_vec(x, category_num):
    vec = np.zeros(category_num)
    vec[x] = 1
    return vec


def make_mask_img(segment_df, category_num, WIDTH, HEIGHT):
    seg_width = segment_df.at[0, "Width"]
    seg_height = segment_df.at[0, "Height"]
    seg_img = np.full(seg_width * seg_height, category_num - 1, dtype=np.int32)
    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
        pixel_list = list(map(int, encoded_pixels.split(" ")))
        for i in range(0, len(pixel_list), 2):
            start_index = pixel_list[i] - 1
            index_len = pixel_list[i + 1] - 1
            seg_img[start_index:start_index + index_len] = int(class_id.split("_")[0])
    seg_img = seg_img.reshape((seg_height, seg_width), order='F')
    seg_img = cv2.resize(seg_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    """
    seg_img_onehot = np.zeros((HEIGHT, WIDTH, category_num), dtype=np.int32)
    #seg_img_onehot = np.zeros((seg_height//ratio, seg_width//ratio, category_num), dtype=np.int32)
    # OPTIMIZE: slow
    for ind in range(HEIGHT):
        for col in range(WIDTH):
            seg_img_onehot[ind, col] = make_onehot_vec(seg_img[ind, col])
    """
    return seg_img


def encode(input_string):
    return [(len(list(g)), k) for k, g in groupby(input_string)]


def run_length(label_vec,category_num):
    encode_list = encode(label_vec)
    index = 1
    class_dict = {}
    for i in encode_list:
        if i[1] != category_num - 1:
            if i[1] not in class_dict.keys():
                class_dict[i[1]] = []
            class_dict[i[1]] = class_dict[i[1]] + [index, i[0]]
        index += i[0]
    return class_dict
