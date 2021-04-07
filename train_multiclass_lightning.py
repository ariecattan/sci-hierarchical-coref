import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import socket
import yaml
from torch.utils import data
from tqdm import tqdm
import os
from transformers import AutoTokenizer
import logging

from utils.utils import *
from models.datasets import CrossEncoderDataset
from models.muticlass import CorefEntailmentLightning




def init_logs():
    root_logger = logging.getLogger()
    logger = root_logger.getChild(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=os.path.join(
        config['log'], '{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return logger



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/multiclass.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger = init_logs()
    logger.info("pid: {}".format(os.getpid()))
    logger.info('Server name: {}'.format(socket.gethostname()))

    fix_seed(config["random_seed"])
    create_folder(config['model_path'])

    logger.info('loading models')
    model = CorefEntailmentLightning(config)

    logger.info('Loading data')
    train = CrossEncoderDataset(config["data"]["training_set"], full_doc=config['full_doc'])
    train_loader = data.DataLoader(train,
                                   batch_size=config["model"]["batch_size"],
                                   shuffle=True,
                                   collate_fn=model.tokenize_batch,
                                   num_workers=16,
                                   pin_memory=True)
    dev = CrossEncoderDataset(config["data"]["dev_set"], full_doc=config['full_doc'])
    dev_loader = data.DataLoader(dev,
                                 batch_size=config["model"]["batch_size"],
                                 shuffle=False,
                                 collate_fn=model.tokenize_batch,
                                 num_workers=16,
                                 pin_memory=True)

    pl_logger = CSVLogger(save_dir=config['model_path'], name='multiclass')
    pl_logger.log_hyperparams(config)
    checkpoint_callback = ModelCheckpoint(monitor='f1_coref', save_top_k=-1)
    trainer = pl.Trainer(gpus=config['gpu_num'],
                         default_root_dir=config['model_path'],
                         accelerator='ddp',
                         max_epochs=config['model']['epochs'],
                         callbacks=[checkpoint_callback],
                         logger=pl_logger,
                         gradient_clip_val=config['model']['gradient_clip'],
                         accumulate_grad_batches=config['model']['gradient_accumulation'],
                         val_check_interval=1.0)


    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=dev_loader)
    # trainer.test(model, test_dataloaders=dev_loader, ckpt_path='best')


    # trainer.test(model, dev_loader, ckpt_path='models/multiclass_long/epoch=2-step=16190.ckpt')
