import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from utils import UnNormalize

import colorlover as cl

colors = cl.scales['4']['qual']['Set3']
labels = np.array(range(1, 5))
# combining into a dictionary
palette = dict(zip(labels, np.array(cl.to_numeric(colors))))


def mask_to_contours(image, mask_layer, color):
    """ converts a mask to contours using OpenCV and draws it on the image
    """
    #     pdb.set_trace()
    mask_layer = cv.convertScaleAbs(mask_layer)
    # https://docs.opencv.org/4.1.0/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv.findContours(mask_layer, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image = cv.drawContours(image, contours, -1, color, 2)

    return image


def visualise_mask(imgs, masks, unnormalize=True):
    """ open an image and draws clear masks, so we don't lose sight of the
        interesting features hiding underneath
    """

    # going through the 4 layers in the last dimension
    # of our mask with shape (256, 1600, 4)
    # we go through each image

    for index in range(imgs.shape[0]):

        img = imgs[index]
        if unnormalize:
            # rev   erse normalize it
            unnorm = UnNormalize()
            img = unnorm(img)

        img = img.numpy().transpose((1, 2, 0))
        mask = masks[index].numpy().transpose((1, 2, 0))

        # fig, axes = plt.subplots(mask.shape[-1], 1, figsize=(16, 20))
        # fig.tight_layout()

        for i in range(mask.shape[-1]):
            # indeces are [0, 1, 2, 3], corresponding classes are [1, 2, 3, 4]
            if np.amax(mask[:, :, i]) > 0.0:
                label = i + 1
                # ax = axes[i]
                # add the contours, layer per layer
                image = mask_to_contours(img, mask[:, :, i], color=palette[label])

                plt.figure(figsize=(16, 30))
                plt.title("In image {} for defect {}".format(index, i))
                plt.imshow(image.get())


def show_mask_image(imgs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), unnormalize=True, masks=None):
    """
    Mask is 4 dimension because we have mask for 4 different classes
    """

    for index in range(imgs.shape[0]):

        img = imgs[index]
        if unnormalize:
            # rev   erse normalize it
            unnorm = UnNormalize(mean=mean, std=std)
            img = unnorm(img)

        img = img.numpy().transpose((1, 2, 0))

        if type(masks) is np.ndarray:
            mask = masks.transpose((1, 2, 0))
        else:
            mask = masks[index].numpy().transpose((1, 2, 0))
        extra = 'Has defect type:'
        fig, ax = plt.subplots(figsize=(15, 50))
        for j in range(4):
            msk = mask[:, :, j]
            if np.sum(msk) != 0: extra += ' ' + str(j + 1)
            if j == 0:  # yellow
                img[msk == 1, 0] = 235
                img[msk == 1, 1] = 235
            elif j == 1:
                img[msk == 1, 0] = 210  # green
            elif j == 2:
                img[msk == 1, 0] = 255  # blue
            elif j == 3:  # magenta
                img[msk == 1, 0] = 255
                img[msk == 1, 2] = 255

        plt.axis('off')
        plt.title(extra)
        plt.imshow(img)
        plt.show()
