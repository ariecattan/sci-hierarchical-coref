import torch
from torch.utils import data
import collections
from itertools import product
import numpy as np
import jsonlines




class CrossEncoderDataset(data.Dataset):
    def __init__(self, data_path, full_doc=True, sep_token='</s>'):
        super(CrossEncoderDataset, self).__init__()
        with jsonlines.open(data_path, 'r') as f:
            self.data = [topic for topic in f]
        self.sep = sep_token
        self.full_doc = full_doc

        self.pairs, self.labels = [], []
        self.info_pairs = []
        for i, topic in enumerate(self.data):
            inputs, labels, info_pairs = self.get_topic_pairs(topic)
            self.pairs.extend(inputs)
            self.labels.extend(labels)
            pair_nums = len(info_pairs)
            info_pairs = np.concatenate((np.array([i] * pair_nums).reshape(pair_nums, 1),
                                        info_pairs), axis=1)
            self.info_pairs.extend(info_pairs)

        self.labels = torch.tensor(self.labels, dtype=torch.long)



    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]


    def get_topic_pairs(self, topic):
        '''
        :param topic:
        :return:
        '''
        relations = [(x, y) for x, y in topic['relations']]
        mentions = []
        for mention in topic['mentions']:
            if self.full_doc:
                mentions.append(self.get_full_doc_mention(mention, topic['tokens']))
            else:
                mentions.append(self.get_sentence_context(mention, topic['tokens'], topic['sentences']))
        mentions = np.array(mentions)

        first, second = zip(*[(x, y) for x, y in product(range(len(mentions)), repeat=2) if x != y])
        first, second = np.array(first), np.array(second)
        seps = np.array([self.sep] * len(first))
        inputs = np.char.add(np.char.add(mentions[first], seps), mentions[second]).tolist()

        labels = []
        for x, y in zip(first, second):
            cluster_x, cluster_y = topic['mentions'][x][-1], topic['mentions'][y][-1]
            if cluster_x == cluster_y:
                labels.append(1)
            elif (cluster_x, cluster_y) in relations:
                labels.append(2)
            elif (cluster_y, cluster_x) in relations:
                labels.append(3)
            else:
                labels.append(0)

        return inputs, labels, list(zip(first, second))



    def get_full_doc_mention(self, mention, tokens):
        doc_id, start, end, _ = mention
        mention_rep = tokens[doc_id][:start] + ['<m>']
        mention_rep += tokens[doc_id][start:end + 1] + ['</m>']
        mention_rep += tokens[doc_id][end + 1:]
        return ' '.join(mention_rep)




    def get_sentence_context(self, mention, tokens, sentences):
        doc_id, start, end, _ = mention
        sent_start, sent_end = 0, len(tokens) - 1
        i = 0
        while i < len(sentences[doc_id]):
            sent_start, sent_end = sentences[doc_id][i]
            if start >= sent_start and end <= sent_end:
                break
            i += 1

        mention_rep = tokens[doc_id][sent_start:start] + ['<m>']
        mention_rep += tokens[doc_id][start:end + 1] + ['</m>']
        mention_rep += tokens[doc_id][end + 1:sent_end] + [self.sep]

        return ' '.join(mention_rep)





class ClusterDataset(data.Dataset):
    def __init__(self, topic, clusters):
        super(ClusterDataset, self).__init__()
        self.topic = topic
        self.clusters = clusters


        '''
        generate all pairs of clusters (product) 
        where each cluster is the concatenation of the all the docs 
        while highlighting the mention        
        '''

