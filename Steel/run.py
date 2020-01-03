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
    argparser.add_argument('--gpu', type=str, help='gpus for training', default='0')
    args = argparser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.configfile)

    if not args.gpu:
        gpus = args.gpu
        config.set('COMMON', 'GPUS', gpus)
    else:
        gpus = config.get('COMMON', 'GPUS')

    encoder = config.get('MODEL', 'ENCODER')
    encoder_weights = config.get('MODEL', 'ENCODER_WEIGHTS')
    data_folder = config.get('FILES', 'DATA_FOLDER')
    train_df_path = config.get('FILES', 'TRAIN_DF_PATH')
    logging_dir = config.get('COMMON', 'LOGGING_DIR')

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    # create a writer
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    TRAIN_STR = time_str + '_' + args.train_tag
    save_model_str = encoder + '_' + TRAIN_STR
    PATH_to_log_dir = logging_dir + '_' + save_model_str
    config.set('MODEL', 'NAME', save_model_str)

    writer = SummaryWriter(PATH_to_log_dir)

    model = Unet(encoder, encoder_weights=encoder_weights, classes=4, activation=None)

    model_trainer = Trainer(model, config=config, writer=writer)
    model_trainer.start()
    writer.close()
