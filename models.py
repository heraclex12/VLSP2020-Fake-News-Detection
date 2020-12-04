from transformers import *
import torch


class RobertaReINTELClassification(BertPreTrainedModel):
    """
        Custom RoBERTa Base Model
    """
    def __init__(self, config):
        super(RobertaReINTELClassification, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.num_labels = config.num_labels
        self.outputs = torch.nn.Linear(config.hidden_size * 4, self.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        cls_output = torch.cat((outputs[2][-1][:, 0, ...], outputs[2][-2][:, 0, ...], outputs[2][-3][:, 0, ...],
                                outputs[2][-4][:, 0, ...]), -1)
        logits = self.outputs(cls_output)
        return logits


class RobertaLargeReINTELClassification(BertPreTrainedModel):
    """
        Custom RoBERTa Large Model
    """
    def __init__(self, config):
        super(RobertaLargeReINTELClassification, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.num_labels = config.num_labels
        self.outputs = torch.nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)[1]
        logits = self.outputs(outputs)
        return logits


class XLMRobertaReINTELClassification(BertPreTrainedModel):
    """
        Custom XLM-RoBERTa Model
    """
    def __init__(self, config):
        super(XLMRobertaReINTELClassification, self).__init__(config)
        self.roberta = XLMRobertaModel(config)
        self.num_labels = config.num_labels
        self.outputs = torch.nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)[1]
        logits = self.outputs(outputs)
        return logits


class ElectraReINTELClassification(ElectraPreTrainedModel):
    """
        Custom ELECTRA Model
    """
    def __init__(self, config):
        super(ElectraReINTELClassification, self).__init__(config=config)
        self.electra = ElectraModel(config)
        self.num_labels = config.num_labels
        self.ln = torch.nn.Linear(config.hidden_size * 4, self.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):
        outputs = \
            self.electra(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask)[1]

        cls_output = torch.cat(
            (outputs[-1][:, 0, ...], outputs[-2][:, 0, ...], outputs[-3][:, 0, ...], outputs[-4][:, 0, ...]), -1)
        logits = self.ln(cls_output)
        return logits


class BertReINTELClassification(BertPreTrainedModel):
    """
        Custom BERT model
    """
    def __init__(self, config):
        super(BertReINTELClassification, self).__init__(config=config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.ln = torch.nn.Linear(config.hidden_size * 4, self.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask)[2]

        cls_output = torch.cat(
            (outputs[-1][:, 0, ...], outputs[-2][:, 0, ...], outputs[-3][:, 0, ...], outputs[-4][:, 0, ...]), -1)
        logits = self.ln(cls_output)
        return logits
