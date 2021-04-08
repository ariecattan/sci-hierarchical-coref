from evaluate import get_average_strict_hypernym, get_all_scores
import numpy as np
import jsonlines
import os
import pandas as pd

from coval.coval.conll import reader
from coval.coval.eval import evaluator

import sys


def evaluate(key_file, sys_file, metrics, NP_only, remove_nested, keep_singletons, min_spans):
    doc_coref_infos = reader.get_coref_infos(key_file, sys_file, NP_only,
                                             remove_nested, keep_singletons, min_spans)

    scores = {}
    conll = 0
    for name, metric in metrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos, metric, beta=1)

        scores['{}_{}'.format(name, 'recall')] = recall
        scores['{}_{}'.format(name, 'precision')] = precision
        scores['{}_{}'.format(name, 'f1')] = f1

        if name in ["muc", "bcub", "ceafe"]:
            conll += f1


    return conll / 3 * 100


def eval_coref(system, annotators, source, gold_path=None):
    allmetrics = [('mentions', evaluator.mentions), ('muc', evaluator.muc),
                  ('bcub', evaluator.b_cubed), ('ceafe', evaluator.ceafe),
                  ('lea', evaluator.lea)]

    NP_only = False
    remove_nested = False
    keep_singletons = True
    min_span = False

    conll_f1 = []



    for annotator in annotators:
        if gold_path is None:
            annotator_path = 'data/test/conll/{}_{}.conll'.format(source, annotator)
        else:
            annotator_path = os.path.join(gold_path, 'conll', '{}_{}.conll'.format(source, annotator))

        conll = evaluate(system, annotator_path, allmetrics, NP_only, remove_nested,
            keep_singletons, min_span)

        conll_f1.append(conll)


    return conll_f1





def eval_hypernym(system, annotators, gold_path=None):
    sys_dic = get_dict_from_json(system)

    f1s = []
    for annotator in annotators:
        if gold_path is None:
            annotator_path = 'data/test/jsonl/{}.jsonl'.format(annotator)
        else:
            annotator_path = os.path.join(gold_path, 'jsonl',  '{}.jsonl'.format(annotator))



        gold_dic = get_dict_from_json(annotator_path)
        _, _, f1 = get_average_strict_hypernym(gold_dic, sys_dic)

        f1s.append(f1 * 100)

    return f1s




def get_dict_from_json(path):
    with jsonlines.open(path, 'r') as f:
        gold = [line for line in f]
    return {x['id']: x for x in gold}





def eval_graph(system, annotators, gold_path=None):
    sys_dic = get_dict_from_json(system)

    directed, undirected = [], []
    for annotator in annotators:
        if gold_path is None:
            annotator_path = 'data/test/jsonl/{}.jsonl'.format(annotator)
        else:
            annotator_path = os.path.join(gold_path, 'jsonl', '{}.jsonl'.format(annotator))

        gold_dic = get_dict_from_json(annotator_path)

        micro_directed, _, _, _ = get_all_scores(gold_dic, sys_dic, directed=True, with_tn=False)
        micro_undirected, _, _, _ = get_all_scores(gold_dic, sys_dic, directed=False, with_tn=False)

        directed.append(micro_directed)
        undirected.append(micro_undirected)

    return directed, undirected



def get_mean_max_std(scores):
    return round(np.mean(scores), 2), \
           round(np.std(scores), 2), \
           round(max(scores), 2)




if __name__ == '__main__':
    dir_pred = sys.argv[1]



    sys_coref = os.path.join(dir_pred, [x for x in os.listdir(dir_pred) if x.endswith('simple.conll')][0])
    sys_connected = os.path.join(dir_pred, [x for x in os.listdir(dir_pred) if x.endswith('connected.conll')][0])
    sys_jsonl = os.path.join(dir_pred, [x for x in os.listdir(dir_pred) if x.endswith('jsonl')][0])

    annotators = ['binu', 'jia', 'irina', 'mostafa']


    all_scores = {}
    # simple coref
    print('Evaluate coref')
    simple_coref = eval_coref(sys_coref, annotators, source='simple')
    mean, std, maxi = get_mean_max_std(simple_coref)
    all_scores['coref'] = [mean, std, maxi]

    # hypernym
    print('Evaluate hypernym')
    strict_hypernym = eval_hypernym(sys_jsonl, annotators)
    mean, std, maxi = get_mean_max_std(strict_hypernym)
    all_scores['hypernym'] = [mean, std, maxi]

    # graph based
    print('Evaluate path based')
    directed, undirected = eval_graph(sys_jsonl, annotators)
    mean, std, maxi = get_mean_max_std(directed)
    all_scores['directed'] = [mean, std, maxi]
    mean, std, maxi = get_mean_max_std(undirected)
    all_scores['undirected'] = [mean, std, maxi]

    # connected components
    print('Evaluate connected components')
    connected = eval_coref(sys_connected, annotators, source='connected')
    mean, std, maxi = get_mean_max_std(connected)
    all_scores['connected'] = [mean, std, maxi]



    df = pd.DataFrame.from_dict(all_scores, orient='index', columns=['mean', 'std', 'max'])

    print(df)