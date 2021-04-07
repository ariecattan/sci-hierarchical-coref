import torch
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics


class BertMNLIFinetuner(pl.LightningModule):
    def __init__(self, config):
        super(BertMNLIFinetuner, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
        self.model = AutoModel.from_pretrained(config.bert_model)
        self.W = nn.Linear(self.model.config.hidden_size, 1)
        self.num_classes = 1
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1(num_classes=1)


    def forward(self, tokens):

        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        h, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        h_cls = h[:, 0, :]
        logits = self.W(h_cls)
        return logits


    def training_step(self, batch, batch_idx):
        # batch
        tokens, label = batch

        # fwd
        y_hat = self(tokens)

        # loss
        loss = self.criterion(y_hat, label)

        # logs
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        # batch
        tokens, label = batch

        # fwd
        y_hat = self(tokens)

        # loss
        loss = self.criterion(y_hat, label)

        # acc, f1
        val_acc = self.accuracy(y_hat, label)
        val_f1 = self.f1(y_hat, label)

        return {'val_loss': loss, 'val_acc': val_acc, 'val_f1': val_f1}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer




