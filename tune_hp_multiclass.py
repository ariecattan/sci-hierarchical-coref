import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import socket
import yaml
from torch.utils import data
import logging
from datetime import datetime
import os
import torch
from tqdm import tqdm
import jsonlines
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import collections
import pandas as pd
from itertools import product

from models.datasets import CrossEncoderDataset
from models.muticlass import CorefEntailmentLightning
from predict_multiclass import MulticlassInference

from eval.shortest_path import ShortestPath





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/multiclass.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

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
    # root_logger.addHandler(file_handler)
    logger.info("pid: {}".format(os.getpid()))
    logger.info('Server name: {}'.format(socket.gethostname()))


    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])


    logger.info('loading models')
    model = CorefEntailmentLightning.load_from_checkpoint(config['checkpoint'], config=config)

    logger.info('Loading data')
    dev = CrossEncoderDataset(config["data"]["dev_set"], full_doc=config['full_doc'])
    dev_loader = data.DataLoader(dev,
                                  batch_size=config["model"]["batch_size"] * 32,
                                  shuffle=False,
                                  collate_fn=model.tokenize_batch,
                                  num_workers=16,
                                  pin_memory=True)

    # pl_logger = CSVLogger(save_dir='logs', name='multiclass_inference')
    # trainer = pl.Trainer(gpus=config['gpu_num'], accelerator='dp')
    # results = trainer.predict(model, dataloaders=dev_loader)
    # results = torch.cat([torch.tensor(x) for x in results])
    # torch.save(results, 'checkpoints/results.pt')

    results = torch.load('checkpoints/preds.pt')
    coref_threshold = np.arange(0.2, 0.81, 0.05)
    hypernym_threshold = np.arange(0.2, 0.81, 0.05)


    ## run predict for all thresholds
    scores = []
    pairs = [(x, y) for x in coref_threshold for y in hypernym_threshold]
    logger.info(f'Predicting {len(pairs)} configurations...')
    for coref, hypernym in tqdm(pairs):
        inference = MulticlassInference(dev, results, coref, hypernym)
        inference.predict_cluster_relations()
        path_based = ShortestPath(dev.data, inference.predicted_data, directed=True, with_tn=True)
        scores.append(path_based.micro_average)


    best = np.argmax(scores)
    logger.info(f'Highest score: {scores[best]}')
    logger.info(f'coref threshold: {pairs[best][0]} hypernym threshold: f{pairs[best][1]}')
    ## evaluate data

