import jsonlines
import argparse
import pyhocon
from datetime import datetime
import torch
import os
from tqdm import tqdm
import numpy as np
from itertools import product
import pytorch_lightning as pl
import networkx as nx
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from utils.corpus import Corpus
from utils.dataset import HypernymDataset
import socket
from models.hypernym_models import BertMNLIFinetuner

import logging
root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)




def get_hypernym_candidates(topic, clusters):
    cluster_ids, candidates = [], []
    for c_id, mentions in clusters.items():
        mention_text = topic['mention_text'][mentions]
        sampled = np.random.choice(mention_text, min(len(mention_text), 10), replace=False)
        candidates.append(', '.join(sampled))
        cluster_ids.append(c_id)

    permutations = list(product(range(len(candidates)), repeat=2))
    permutations = [(a, b) for a, b in permutations if a != b]
    labels = [1  if [b, a] in topic['relations'] else 0 for a, b in permutations]
    labels = torch.tensor(labels)
    first, second = zip(*permutations)
    first, second = torch.tensor(first), torch.tensor(second)

    return candidates, first, second, labels





def fine_tune_entailment_model(topic, clusters, entailment_model, device, batch_size=32):
    candidates, first, second, labels = get_hypernym_candidates(topic, clusters)
    labels = labels.to(torch.float).to(device)
    candidates = np.array(candidates)
    premise = candidates[first]
    hypothesis = candidates[second]

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(entailment_model.parameters(), lr=config.learning_rate)

    for i in tqdm(range(0, len(labels), batch_size)):
        optimizer.zero_grad()
        maxi = min(i + batch_size, len(labels))
        batch_premise = premise[i:maxi]
        batch_hypothesis = hypothesis[i:maxi]
        batch_labels = labels[i:maxi]
        scores, predictions = entailment_model(batch_premise, batch_hypothesis)

        loss = criterion(scores, batch_labels)
        loss.backward()
        optimizer.step()


    return entailment_model



def entailment_decoding(cluster_ids, first, second, scores, predictions):
    positives = torch.nonzero(predictions == 2).squeeze(-1)
    scores = scores[positives]
    first, second = first[positives], second[positives]

    graph = nx.DiGraph(directed=False)
    graph.add_nodes_from(cluster_ids)

    _, indices = torch.sort(scores, descending=True)
    for ind in zip(indices):
        parent, child = cluster_ids[second[ind]], cluster_ids[first[ind]]

        ind = sum([graph.has_edge(node, child) for node in graph.nodes])
        if ind == 0:  # to prevent multiple parents to the same node
            graph.add_edge(parent, child)
            if len(list(nx.simple_cycles(graph))) > 0:  # to prevent cycle
                graph.remove_edge(parent, child)


    return [[int(a), int(b)] for a, b in graph.edges]




def evaluate_entailment_model(topic, clusters, entailment_model):
    entailment_model.eval()

    candidates, first, second, labels = get_hypernym_candidates(topic, clusters)
    candidates = np.array(candidates)
    premise = candidates[first]
    hypothesis = candidates[second]

    with torch.no_grad():
        scores, predictions = entailment_model(premise, hypothesis)


    # edges = entailment_decoding(list(clusters.keys()), first, second, scores, predictions)








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/hypernym.yaml')
    args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file(args.config)
    # with open(args.config, 'r') as f:
    #     config = yaml.safe_load(f)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info(config)
    logger.info('{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))
    logger.info('using server {}'.format(socket.gethostname()))
    logger.info("pid: {}".format(os.getpid()))

    device = 'cuda:{}'.format(config['gpu_num'][0]) if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(config.bert_model)

    logger.info('Loading data..')
    with jsonlines.open(config['training_set'], 'r') as f:
        train = [line for line in f]
    train = Corpus(train, tokenizer=None)
    train_dataset = HypernymDataset(train, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    with jsonlines.open(config['dev_set'], 'r') as f:
        dev = [line for line in f]
    dev = Corpus(dev, tokenizer=None)
    dev_dataset = HypernymDataset(dev, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size  * 4, shuffle=False)

    logger.info('Training')
    entailment_model = BertMNLIFinetuner(config)
    trainer = pl.Trainer(gpus=4, max_epochs=3)
    trainer.fit(entailment_model, train_dataloader=train_loader, val_dataloaders=dev_loader)




    #
    # for topic_num, topic in enumerate(train.topics):
    #     logger.info(f'processing topic {topic["id"]}')
    #     mentions = np.array(topic['mentions'])
    #     clusters = mentions[:, -1]
    #
    #     all_clusters = collections.defaultdict(list)
    #     for i, cluster_id in enumerate(clusters):
    #         all_clusters[cluster_id].append(i)
    #
    #     fine_tune_entailment_model(topic, all_clusters, entailment_model, device, config.batch_size)
    #
    #
    #
