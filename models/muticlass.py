import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
import pytorch_lightning as pl
import torchmetrics

from typing import Any, List, Optional

class CorefEntailmentLightning(pl.LightningModule):
    '''
    multiclass classification with labels:
    0 not related
    1 coref
    2. hypernym
    3. neutral (hyponym)
    '''

    def __init__(self, config, mode='dpp'):
        super(CorefEntailmentLightning, self).__init__()
        self.long = True if 'longformer' in config["model"]["bert_model"] else False
        self.config = config


        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')

        self.model = AutoModel.from_pretrained(config["model"]["bert_model"], add_pooling_layer=False)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.criterion = torch.nn.CrossEntropyLoss()


        self.val_acc = pl.metrics.Accuracy(top_k=1)
        self.val_f1 = pl.metrics.F1(num_classes=4, average='none')
        self.val_recall = pl.metrics.Recall(num_classes=4, average='none')
        self.val_precision = pl.metrics.Precision(num_classes=4, average='none')


    def forward(self, input_ids, attention_mask, global_attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask,
                            global_attention_mask=global_attention_mask)
        cls_vector = output.last_hidden_state[:, 0, :]
        scores = self.linear(cls_vector)
        return scores



    def training_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        return loss



    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        y_hat = torch.softmax(y_hat, dim=1)
        loss = self.criterion(y_hat, y)

        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_precision(y_hat, y)

        self.log('val_loss', loss, on_epoch=True, on_step=False)

        return loss


    def validation_epoch_end(self, outputs):
        self.log_metrics()



    #
    def test_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        y_hat = torch.softmax(y_hat, dim=1)
        loss = self.criterion(y_hat, y)
        # self.val_acc(y_hat, y)
        # self.val_f1(y_hat, y)
        # self.val_recall(y_hat, y)
        # self.val_precision(y_hat, y)

        return {
            'loss': loss,
            'preds': y_hat,
            'label': y
        }


    def test_step_end(self, outputs):
        y_hat, y = outputs['preds'], outputs['label']
        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_precision(y_hat, y)

        return outputs


    def test_epoch_end(self, outputs):
        self.log_metrics()
        self.results = outputs

        return torch.cat([x['preds'] for x in outputs])

        # return outputs
        # preds = torch.cat([x['preds'] for x in outputs])
        # self.results = preds


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        y_hat = torch.softmax(y_hat, dim=1)
        return y_hat









    def log_metrics(self):
        self.log('val_acc', self.val_acc.compute())
        f1_negative, f1_coref, f1_hypernym, f1_hyponym = self.val_f1.compute()
        recall_negative, recall_coref, recall_hypernym, recall_hyponym = self.val_recall.compute()
        precision_negative, precision_coref, precision_hypernym, precision_hyponym = self.val_precision.compute()
        self.log('f1_coref', f1_coref)
        self.log('recall_coref', recall_coref)
        self.log('precision_coref', precision_coref)
        self.log('f1_hypernym', f1_hypernym)
        self.log('recall_hypernym', recall_hypernym)
        self.log('precision_hypernym', precision_hypernym)
        self.log('f1_hyponym', f1_hyponym)
        self.log('recall_hyponym', recall_hyponym)
        self.log('precision_hyponym', precision_hyponym)



    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])


    def get_global_attention(self, input_ids):
        global_attention_mask = torch.zeros(input_ids.shape)
        global_attention_mask[:, 0] = 1  # global attention to the CLS token
        start = torch.nonzero(input_ids == self.start)
        end = torch.nonzero(input_ids == self.end)
        globs = torch.cat((start, end))
        value = torch.ones(globs.shape[0])
        global_attention_mask.index_put_(tuple(globs.t()), value)
        return global_attention_mask


    def tokenize_batch(self, batch):
        inputs, labels = zip(*batch)
        tokens = self.tokenizer(list(inputs), padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        global_attention_mask = self.get_global_attention(input_ids)
        labels = torch.stack(labels)

        return (input_ids, attention_mask, global_attention_mask), labels