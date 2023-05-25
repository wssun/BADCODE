import torch
import torch.nn as nn
import numpy as np
from transformers import (RobertaTokenizer, T5Config, T5ForConditionalGeneration)
import logging

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)}


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x, **kwargs):
        # x = x.reshape(-1, x.size(-1) * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class SearchModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(SearchModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec, hidden_states

    def forward(self, source_ids=None, labels=None):
        # source_ids = source_ids.view(-1, self.args.max_source_length)

        vec, _ = self.get_t5_vec(source_ids)

        logits = self.classifier(vec)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

    def output_t5_vec(self, source_ids=None):
        vec, _ = self.get_t5_vec(source_ids)
        return vec

    def output_hidden_states(self, source_ids=None):
        _, hidden_states = self.get_t5_vec(source_ids)
        return hidden_states
