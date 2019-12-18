import torch.backends.cudnn as cudnn
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from logging_metric import Meter
from torch.utils.tensorboard import SummaryWriter
from train import epoch_log
from dataset import provider
import torch.nn as nn

import time
import datetime


class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model, writer, config):
        self.writer = writer
        self.data_folder = config.get('FILES', 'DATA_FOLDER')
        self.train_df_path = config.get('FILES', 'TRAIN_DF_PATH')
        self.save_model_dir = config.get('FILES', 'TRAINED_MODELS_DIR')
        self.num_workers = config.getint('COMMON', 'NO_WORKERS')
        self.batch_train = config.getint('TRAINING', 'BATCH_TRAIN')
        self.batch_val = config.getint('TRAINING', 'BATCH_VAL')
        self.batch_size = {"train": self.batch_train, "val": self.batch_val}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = config.getfloat('TRAINING', 'LR')
        self.num_epochs = config.getint('TRAINING', 'EPOCH')
        self.mean = config.get('PROCESSING', 'MEAN').split(',')
        self.std = config.get('PROCESSING', 'STD').split(',')
        self.seed = config.getint('COMMON', 'SEED')

        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.net = self.net.to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = nn.DataParallel(self.net)

        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                data_folder=self.data_folder,
                df_path=self.train_df_path,
                phase=phase,
                mean=self.mean,
                std=self.std,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
                seed=self.seed
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch, self.writer)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        # batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):  # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        # update board
        meter.update_board(epoch_loss, dice, iou, images, targets)

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                torch.save(state, self.save_model_dir + time_str + '_model.pth')
            print()
