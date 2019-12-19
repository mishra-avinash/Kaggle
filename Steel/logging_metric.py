import torch
from utils import metric
from train import predict, compute_iou_batch
import numpy as np
from utils import UnNormalize


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''

    def __init__(self, phase, epoch, writer):
        self.base_threshold = 0.5  # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []
        self.phase = phase
        self.epoch = epoch
        self.writer = writer

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def update_board(self, loss, dice, iou, images, masks):
        # update the tensorboard
        if self.phase == 'train':
            name_str = 'Train/'
        else:
            name_str = 'Val/'
        self.writer.add_scalar(tag=name_str + "Loss", scalar_value=loss, global_step=self.epoch)
        self.writer.add_scalar(tag=name_str + "Dice", scalar_value=dice, global_step=self.epoch)
        self.writer.add_scalar(tag=name_str + "Iou", scalar_value=iou, global_step=self.epoch)
        # unnormalize images before adding to board

        for index in range(images.shape[0]):
            img = images[index]
            unnorm = UnNormalize()
            img = unnorm(img)
            self.writer.add_image(tag=name_str + "Img", img_tensor=img, global_step=self.epoch, dataformats='CHW')
        self.writer.flush()
        # self.writer.add_images(tag=name_str + "Masks", img_tensor=masks, global_step=self.epoch, dataformats='NCHW')

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou
