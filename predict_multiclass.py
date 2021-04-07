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
from itertools import product

from models.datasets import CrossEncoderDataset
from models.muticlass import CorefEntailmentLightning
from utils.model_utils import get_greedy_relations


class MulticlassInference:
    def __init__(self, dataset, pairwise_scores, coref_threshold, hypernym_threshold):
        self.dataset = dataset
        self.info_pairs = torch.tensor(dataset.info_pairs)
        self.pairwise_scores = pairwise_scores
        self.coref_threshold = coref_threshold
        self.hypernym_threshold = hypernym_threshold


        self.clustering = AgglomerativeClustering(n_clusters=None,
                                                  affinity='precomputed',
                                                  linkage='average',
                                                  distance_threshold=self.coref_threshold)



    def get_coref_adjacency_matrix(self, info_pairs, pairwise_scores):
        '''
        make coreference adjacency matrix
        :param info_pairs:
        :param pairwise_scores:
        :return: adjacency matrix of coref
        '''
        num_of_mentions = info_pairs.max().item() + 1
        adjacency = torch.eye(num_of_mentions)
        coref_predictions = torch.argmax(pairwise_scores, dim=1)
        preds = torch.nonzero(coref_predictions != 0).squeeze(-1)
        pairs = info_pairs[preds][:, 1:]
        coref_scores = pairwise_scores[preds][:, 1]
        adjacency.index_put_(tuple(pairs.t()), coref_scores)

        return adjacency.numpy()


    def undirect_adjacency_matrix(self, matrix):
        undirected = matrix.copy()
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                maxi = max(matrix[i, j], matrix[j, i])
                undirected[i, j] = maxi
                undirected[j, i] = maxi

        return undirected

    def get_mention_pairs(self, cluster_a, cluster_b, pairs, scores):
        cluster_pairs = [[x, y] for x in cluster_a for y in cluster_b]
        indices = [i for i, x in enumerate(pairs) if x in cluster_pairs]
        return scores[indices].mean()


    def hypernym_relations(self, predicted_mentions, pairs, scores):
        '''
        return relations between clusters hypernym scores=2, hyponym=3
        :param clusters:
        :param pairs:
        :param scores:
        :return:
        '''

        clusters = collections.defaultdict(list)
        for i, (_, _, _, c_id) in enumerate(predicted_mentions):
            clusters[c_id].append(i)

        cluster_ids = list(clusters.keys())
        permutations = list(product(range(len(cluster_ids)), repeat=2))
        permutations = [(x, y) for x, y in permutations if x != y]
        first, second = zip(*permutations)
        info_pairs = pairs[:, 1:].tolist()
        avg_scores = torch.stack([self.get_mention_pairs(clusters[x], clusters[y], info_pairs, scores[:, 3])
                                  for x, y in zip(first, second)])

        inds = torch.nonzero(avg_scores >= self.hypernym_threshold).squeeze(-1)
        avg_scores = avg_scores[inds]
        first, second = torch.tensor(first)[inds], torch.tensor(second)[inds]

        relations = get_greedy_relations(cluster_ids, avg_scores, first, second)

        return relations




    def get_topic_prediction(self, topic_num, topic_pairs, topic_scores):
        # coref clusters
        coref_adjacency_matrix = self.get_coref_adjacency_matrix(topic_pairs, topic_scores)
        undirected_matrix = self.undirect_adjacency_matrix(coref_adjacency_matrix)
        distance_matrix = 1 - undirected_matrix
        predicted = self.clustering.fit(distance_matrix)


        mentions = self.dataset.data[topic_num]['mentions']
        mentions = np.array(mentions)
        predicted_clusters = predicted.labels_.reshape(len(mentions), 1)
        predicted_mentions = np.concatenate((mentions[:, :-1], predicted_clusters), axis=1)


        relations = self.hypernym_relations(predicted_mentions, topic_pairs, topic_scores)

        return {
            "id": self.dataset.data[topic_num]['id'],
            "tokens":self.dataset.data[topic_num]['tokens'],
            "mentions": predicted_mentions.tolist(),
            "relations": relations
        }


    def predict_cluster_relations(self):
        idx, vals = torch.unique(self.info_pairs[:, 0], return_counts=True)
        all_pairs = torch.split_with_sizes(self.info_pairs, tuple(vals))
        all_scores = torch.split_with_sizes(self.pairwise_scores, tuple(vals))

        predicted_data = []
        for topic, (topic_pair, topic_scores) in enumerate(zip(all_pairs, all_scores)):
            data = self.get_topic_prediction(topic, topic_pair, topic_scores)
            predicted_data.append(data)

        self.predicted_data = predicted_data


    def save_predicted_file(self, output_dir):
        jsonl_path = os.path.join(output_dir, 'system_{}_{}.jsonl'.format(self.coref_threshold, self.hypernym_threshold))
        with jsonlines.open(jsonl_path, 'w') as f:
            f.write_all(self.predicted_data)


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
    root_logger.addHandler(file_handler)
    logger.info("pid: {}".format(os.getpid()))
    logger.info('Server name: {}'.format(socket.gethostname()))

    logger.info('loading models')
    model = CorefEntailmentLightning.load_from_checkpoint(config['checkpoint'], config=config)

    logger.info('Loading data')
    test = CrossEncoderDataset(config["data"]["test_set"], full_doc=config['full_doc'])
    test_loader = data.DataLoader(test,
                                 batch_size=config["model"]["batch_size"] * 32,
                                 shuffle=False,
                                 collate_fn=model.tokenize_batch,
                                 num_workers=16,
                                 pin_memory=True)

    logger = CSVLogger(save_dir='logs', name='multiclass_inference')
    trainer = pl.Trainer(gpus=config['gpu_num'], accelerator='dp', logger=logger)
    results = trainer.predict(model, dataloaders=test_loader)
    results = torch.cat([torch.tensor(x) for x in results])


    inference = MulticlassInference(test, results, config['agg_threshold'], config['hypernym_threshold'])
    inference.predict_cluster_relations()
    inference.save_predicted_file(config['save_path'])