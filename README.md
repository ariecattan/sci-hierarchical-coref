



## Citation 

## Data
 

## Training 

1. Pipeline: coref model (Cattan et al., 2020) and entailement model for relations.
    * Training the coref model `python train_coref_scorer.py --configs configs/coref.yaml` 
    * Find the best threshold of the agglomerative clustering on the dev set `python tuned_threshold.py --config configs/config_clustering.json`
    * Fine-tune the entailment model 

2. Multiclass
    * Training the multiclass model `python train_multiclass.py --config configs/multiclass.yaml`
    * Find the best threshold for the agglomerative clustering and the hypernym relations
     `python tune_hp_multiclass.py --configs configs/mutliclass.yaml`



## Inference


* Boundary baseline (all singletons and all in the same cluster) + entailment for the relations: 

```bash
python boundary_baseline.py --gpu 0 \
    --data_path data/test/jsonl/binu.jsonl \
    --output_dir checkpoints/boundary \
    --nli_model roberta-large-mnli
```


*  Cosine Similarity + entailment model:  

```
python predict_cosine_similarity.py --gpu 0 \
    --data_path data/test/jsonl/binu.jsonl \
    --output_dir checkpoints/boundary \
    --bert_model bert-large-cased \
    --nli_model roberta-large-mnli \
    --threshold 0.5 
```

* Cross-document coref model (Cattan et al, 2020) + entailment model 


```bash
python predict.py 
```


* Multiclass model (set in the `configs/multiclass.yaml` the path to `checkpoint`, the threshold for the agglomerative clustering 
`agg_threshold` and the threshold for the hypernym relations `hypernym_threshold`)
```
python predict_multiclass.py --config configs/multiclass.yaml
```


## Evaluation 

The inference script saves a `.jsonl` file which needs to be evaluated against the `test.jsonl` for all evaluation metrics.

```
python eval/evaluate.py gold system
```