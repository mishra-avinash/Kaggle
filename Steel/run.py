import sys
import configparser
import argparse
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

from unetmodelscript.model import Unet
from trainer_multi import Trainer

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Weather connector')
    argparser.add_argument('--configfile', type=str, default='config.ini', help='Config file (ini format)')
    argparser.add_argument('--train_tag', type=str, help='tag for training')
    args = argparser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.configfile)

    gpus = config.get('COMMON', 'GPUS')
    encoder = config.get('MODEL', 'ENCODER')
    encoder_weights = config.get('MODEL', 'ENCODER_WEIGHTS')
    data_folder = config.get('FILES', 'DATA_FOLDER')
    train_df_path = config.get('FILES', 'TRAIN_DF_PATH')

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    # create a writer
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    TRAIN_STR = time_str + args.train_tag
    PATH_to_log_dir = '../logging/' + TRAIN_STR

    writer = SummaryWriter(PATH_to_log_dir, comment='args.train_tag')

    model = Unet(encoder, encoder_weights=encoder_weights, classes=4, activation=None)

    model_trainer = Trainer(model, config=config, writer=writer)
    model_trainer.start()
