import logging
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.nn import LSTM
from transformers import *
from long_doc_process import *


class DocumentBertLSTM(BertPreTrainedModel):
    """
    BERT output over document in LSTM
    """

    def __init__(self, bert_model_config: BertConfig, args):
        super(DocumentBertLSTM, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size = args['bert_batch_size']
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.lstm = LSTM(bert_model_config.hidden_size, bert_model_config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
        )

    # input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda'):

        # contains all BERT sequences
        # bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                        min(document_batch.shape[1], self.bert_batch_size),
                                        self.config.hidden_size), dtype=torch.float, device=device)

        # only pass through bert_batch_size numbers of inputs into bert.
        # this means that we are possibly cutting off the last part of documents.

        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(
                self.bert(document_batch[doc_id][:self.bert_batch_size, 0],
                          token_type_ids=document_batch[doc_id][:self.bert_batch_size, 1],
                          attention_mask=document_batch[doc_id][:self.bert_batch_size, 2])[1])

        output, (_, _) = self.lstm(bert_output.permute(1, 0, 2))

        last_layer = output[-1]

        prediction = self.classifier(last_layer)

        assert prediction.shape[0] == document_batch.shape[0]
        return prediction

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True


class DocumentElectraLSTM(ElectraPreTrainedModel):
    """
    Electra output over document in LSTM
    """

    def __init__(self, bert_model_config: ElectraConfig, args):
        super(DocumentElectraLSTM, self).__init__(bert_model_config)
        self.bert = ElectraModel(bert_model_config)
        self.bert_batch_size = args['bert_batch_size']
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.lstm = LSTM(bert_model_config.hidden_size, bert_model_config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
        )

    # input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda'):

        # contains all BERT sequences
        # bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                        min(document_batch.shape[1], self.bert_batch_size),
                                        self.config.hidden_size), dtype=torch.float, device=device)

        # only pass through bert_batch_size numbers of inputs into bert.
        # this means that we are possibly cutting off the last part of documents.

        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(
                self.bert(document_batch[doc_id][:self.bert_batch_size, 0],
                          token_type_ids=document_batch[doc_id][:self.bert_batch_size, 1],
                          attention_mask=document_batch[doc_id][:self.bert_batch_size, 2])[0][:, 0, ...])

        output, (_, _) = self.lstm(bert_output.permute(1, 0, 2))

        last_layer = output[-1]

        prediction = self.classifier(last_layer)

        assert prediction.shape[0] == document_batch.shape[0]
        return prediction

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True


class DocumentRobertaLSTM(BertPreTrainedModel):
    """
    Roberta output over document in LSTM
    """

    def __init__(self, bert_model_config: RobertaConfig, args):
        super(DocumentRobertaLSTM, self).__init__(bert_model_config)
        # if args['bert_model_path'].startswith('vinai'):
        self.bert = RobertaModel(bert_model_config)
        # else:
        # self.bert = XLMRobertaModel(bert_model_config)
        self.bert_batch_size = args['bert_batch_size']
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.lstm = LSTM(bert_model_config.hidden_size, bert_model_config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
        )

    # input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda'):

        # contains all BERT sequences
        # bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                        min(document_batch.shape[1], self.bert_batch_size),
                                        self.config.hidden_size), dtype=torch.float, device=device)

        # only pass through bert_batch_size numbers of inputs into bert.
        # this means that we are possibly cutting off the last part of documents.

        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(
                self.bert(document_batch[doc_id][:self.bert_batch_size, 0],
                          token_type_ids=document_batch[doc_id][:self.bert_batch_size, 1],
                          attention_mask=document_batch[doc_id][:self.bert_batch_size, 2])[1])

        output, (_, _) = self.lstm(bert_output.permute(1, 0, 2))

        last_layer = output[-1]

        prediction = self.classifier(last_layer)

        assert prediction.shape[0] == document_batch.shape[0]
        return prediction

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True


document_bert_architectures = {
    'DocumentBertLSTM': DocumentBertLSTM,
    'DocumentElectraLSTM': DocumentElectraLSTM,
    'DocumentRobertaLSTM': DocumentRobertaLSTM,
}


class BertForDocumentClassification(object):
    def __init__(self, args=None,
                 labels=None,
                 device='cuda',
                 bert_model_path='FPTAI/vibert-base-cased',
                 #  bert_model_path='bert-base-multilingual-cased',
                 architecture="DocumentBertLSTM",
                 batch_size=10,
                 bert_batch_size=7,
                 learning_rate=5e-5,
                 weight_decay=0):
        if args is not None:
            self.args = vars(args)
        if not args:
            self.args = {}
            self.args['bert_model_path'] = bert_model_path
            self.args['device'] = device
            self.args['learning_rate'] = learning_rate
            self.args['weight_decay'] = weight_decay
            self.args['batch_size'] = batch_size
            self.args['num_labels'] = 1
            self.args['bert_batch_size'] = bert_batch_size
            self.args['architecture'] = architecture
            self.args['epochs'] = 10

        if 'fold' not in self.args:
            self.args['fold'] = 0

        self.log = logging.getLogger()
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['bert_model_path'], do_lower_case=False)

        config = BertConfig.from_pretrained(self.args['bert_model_path'], num_labels=self.args['num_labels'],
                                            from_tf=False)

        self.bert_doc_classification = document_bert_architectures[self.args['architecture']].from_pretrained(
            self.args['bert_model_path'],
            config=config, args=self.args, from_tf=False)
        self.bert_doc_classification.freeze_bert_encoder()
        self.bert_doc_classification.unfreeze_bert_encoder_last_layers()

        self.optimizer = torch.optim.Adam(
            self.bert_doc_classification.parameters(),
            weight_decay=self.args['weight_decay'],
            lr=self.args['learning_rate']
        )

    def fit(self, train, dev):
        """
        A list of
        :param dev:
        :param train:
        :return:
        """

        train_documents, train_labels = train
        if dev is not None:
            dev_documents, dev_labels = dev

        document_representations, document_sequence_lengths = encode_documents(train_documents, self.bert_tokenizer)

        correct_output = torch.FloatTensor(train_labels)

        loss_weight = ((correct_output.shape[0] / torch.sum(correct_output, dim=0)) - 1).to(device=self.args['device'])
        self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)

        assert document_representations.shape[0] == correct_output.shape[0]

        if torch.cuda.device_count() > 1:
            self.bert_doc_classification = torch.nn.DataParallel(self.bert_doc_classification)
        self.bert_doc_classification.to(device=self.args['device'])
        self.best_score = 0.0
        for epoch in range(1, self.args['epochs'] + 1):
            # shuffle
            self.bert_doc_classification.train()
            permutation = torch.randperm(document_representations.shape[0])
            document_representations = document_representations[permutation]
            document_sequence_lengths = document_sequence_lengths[permutation]
            correct_output = correct_output[permutation]

            self.epoch = epoch
            epoch_loss = 0.0
            for i in range(0, document_representations.shape[0], self.args['batch_size']):
                batch_document_tensors = document_representations[i:i + self.args['batch_size']].to(
                    device=self.args['device'])
                batch_document_sequence_lengths = document_sequence_lengths[i:i + self.args['batch_size']]

                batch_predictions = self.bert_doc_classification(batch_document_tensors,
                                                                 batch_document_sequence_lengths,
                                                                 device=self.args['device'])

                batch_correct_output = correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])
                loss = self.loss_function(batch_predictions, batch_correct_output.unsqueeze(-1))
                epoch_loss += float(loss.item())

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss /= int(
                document_representations.shape[0] / self.args['batch_size'])  # divide by number of batches per epoch

            self.log.info('Epoch %i Completed: %f' % (epoch, epoch_loss))

            if dev is not None:
                self.predict((dev_documents, dev_labels))
                # self.save_checkpoint("trained_models/vbert")


    def predict(self, data, threshold=0.5):
        """
        A tuple containing
        :param threshold:
        :param data:
        :return:
        """
        document_representations = None
        document_sequence_lengths = None
        correct_output = None
        if isinstance(data, list):
            document_representations, document_sequence_lengths = encode_documents(data, self.bert_tokenizer)
        if isinstance(data, tuple) and len(data) == 2:
            self.log.info('Evaluating on Epoch %i' % (self.epoch))
            document_representations, document_sequence_lengths = encode_documents(data[0], self.bert_tokenizer)
            correct_output = torch.FloatTensor(data[1]).unsqueeze(-1).transpose(0, 1)

        self.bert_doc_classification.to(device=self.args['device'])
        self.bert_doc_classification.eval()
        with torch.no_grad():
            predictions = torch.empty((document_representations.shape[0], self.args['num_labels']))
            for i in range(0, document_representations.shape[0], self.args['batch_size']):
                batch_document_tensors = document_representations[i:i + self.args['batch_size']].to(
                    device=self.args['device'])
                batch_document_sequence_lengths = document_sequence_lengths[i:i + self.args['batch_size']]

                prediction = self.bert_doc_classification(batch_document_tensors,
                                                          batch_document_sequence_lengths, device=self.args['device'])

                prediction = torch.sigmoid(prediction)
                predictions[i:i + self.args['batch_size']] = prediction

        if correct_output is None:
            return predictions.transpose(0, 1).cpu()
        else:
            from sklearn.metrics import roc_auc_score
            roc_score = roc_auc_score(correct_output.reshape(-1).numpy(),
                                      predictions.transpose(0, 1).cpu().reshape(-1).numpy())
            if self.best_score < roc_score:
                self.best_score = roc_score
                self.save_checkpoint("trained_models/phobert_base")

            for r in range(0, predictions.shape[0]):
                for c in range(0, predictions.shape[1]):
                    if predictions[r][c] > threshold:
                        predictions[r][c] = 1
                    else:
                        predictions[r][c] = 0
            predictions = predictions.transpose(0, 1)
            assert correct_output.shape == predictions.shape
            precisions = []
            recalls = []
            fmeasures = []

            for label_idx in range(predictions.shape[0]):
                correct = correct_output[label_idx].cpu().view(-1).numpy()
                predicted = predictions[label_idx].cpu().view(-1).numpy()
                present_f1_score = f1_score(correct, predicted, average='binary', pos_label=1)
                present_precision_score = precision_score(correct, predicted, average='binary', pos_label=1)
                present_recall_score = recall_score(correct, predicted, average='binary', pos_label=1)

                precisions.append(present_precision_score)
                recalls.append(present_recall_score)
                fmeasures.append(present_f1_score)

            micro_f1 = f1_score(correct_output.reshape(-1).numpy(), predictions.reshape(-1).numpy(), average='micro')

            print(micro_f1, roc_score)
        self.bert_doc_classification.train()

    def save_checkpoint(self, checkpoint_path: str):
        """
        Saves an instance of the current model to the specified path.
        """
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        self.log.info("Saving checkpoint: %s" % checkpoint_path)

        # save finetune parameters
        net = self.bert_doc_classification
        if isinstance(self.bert_doc_classification, nn.DataParallel):
            net = self.bert_doc_classification.module
        torch.save(net.state_dict(), os.path.join(checkpoint_path, 'pytorch_model.bin'))
        # save configurations
        net.config.to_json_file(os.path.join(checkpoint_path, 'config.json'))
        # save exact vocabulary utilized
        self.bert_tokenizer.save_vocabulary(checkpoint_path)


class ElectraForDocumentClassification(object):
    def __init__(self, args=None,
                 labels=None,
                 device='cuda',
                 bert_model_path='FPTAI/velectra-base-discriminator-cased',
                 architecture="DocumentElectraLSTM",
                 batch_size=10,
                 bert_batch_size=7,
                 learning_rate=5e-5,
                 weight_decay=0):
        if args is not None:
            self.args = vars(args)
        if not args:
            self.args = {}
            self.args['bert_model_path'] = bert_model_path
            self.args['device'] = device
            self.args['learning_rate'] = learning_rate
            self.args['weight_decay'] = weight_decay
            self.args['batch_size'] = batch_size
            self.args['num_labels'] = 1
            self.args['bert_batch_size'] = bert_batch_size
            self.args['architecture'] = architecture
            self.args['epochs'] = 20

        if 'fold' not in self.args:
            self.args['fold'] = 0

        self.log = logging.getLogger()
        self.bert_tokenizer = ElectraTokenizer.from_pretrained(self.args['bert_model_path'], do_lower_case=False)

        config = ElectraConfig.from_pretrained(self.args['bert_model_path'], num_labels=self.args['num_labels'])

        self.bert_doc_classification = document_bert_architectures[self.args['architecture']].from_pretrained(
            self.args['bert_model_path'],
            config=config, args=self.args, from_tf=False)
        self.bert_doc_classification.freeze_bert_encoder()
        self.bert_doc_classification.unfreeze_bert_encoder_last_layers()

        self.optimizer = torch.optim.Adam(
            self.bert_doc_classification.parameters(),
            weight_decay=self.args['weight_decay'],
            lr=self.args['learning_rate']
        )

    def fit(self, train, dev):
        """
        A list of
        :param train:
        :param dev:
        :return:
        """

        train_documents, train_labels = train
        if dev is not None:
            dev_documents, dev_labels = dev

        document_representations, document_sequence_lengths = encode_documents(train_documents, self.bert_tokenizer)

        correct_output = torch.FloatTensor(train_labels)

        loss_weight = ((correct_output.shape[0] / torch.sum(correct_output, dim=0)) - 1).to(device=self.args['device'])
        self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)

        assert document_representations.shape[0] == correct_output.shape[0]

        if torch.cuda.device_count() > 1:
            self.bert_doc_classification = torch.nn.DataParallel(self.bert_doc_classification)
        self.bert_doc_classification.to(device=self.args['device'])

        for epoch in range(1, self.args['epochs'] + 1):
            # shuffle
            self.bert_doc_classification.train()
            permutation = torch.randperm(document_representations.shape[0])
            document_representations = document_representations[permutation]
            document_sequence_lengths = document_sequence_lengths[permutation]
            correct_output = correct_output[permutation]

            self.epoch = epoch
            epoch_loss = 0.0
            for i in range(0, document_representations.shape[0], self.args['batch_size']):
                batch_document_tensors = document_representations[i:i + self.args['batch_size']].to(
                    device=self.args['device'])
                batch_document_sequence_lengths = document_sequence_lengths[i:i + self.args['batch_size']]

                batch_predictions = self.bert_doc_classification(batch_document_tensors,
                                                                 batch_document_sequence_lengths,
                                                                 device=self.args['device'])

                batch_correct_output = correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])
                loss = self.loss_function(batch_predictions, batch_correct_output.unsqueeze(-1))
                epoch_loss += float(loss.item())

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss /= int(
                document_representations.shape[0] / self.args['batch_size'])  # divide by number of batches per epoch

            self.log.info('Epoch %i Completed: %f' % (epoch, epoch_loss))

            if epoch % 2 == 0 or epoch == self.args['epochs']:
                if dev is not None:
                    self.predict((dev_documents, dev_labels))
                torch.save(self.bert_doc_classification.state_dict(), f"model.bin")
                self.save_checkpoint("trained_models/electra")

    def predict(self, data, threshold=0.5):
        """
        A tuple containing
        :param threshold:
        :param data:
        :return:
        """
        document_representations = None
        document_sequence_lengths = None
        correct_output = None
        if isinstance(data, list):
            document_representations, document_sequence_lengths = encode_documents(data, self.bert_tokenizer)
        if isinstance(data, tuple) and len(data) == 2:
            self.log.info('Evaluating on Epoch %i' % (self.epoch))
            document_representations, document_sequence_lengths = encode_documents(data[0], self.bert_tokenizer)
            correct_output = torch.FloatTensor(data[1]).unsqueeze(-1).transpose(0, 1)

        self.bert_doc_classification.to(device=self.args['device'])
        self.bert_doc_classification.eval()
        with torch.no_grad():
            predictions = torch.empty((document_representations.shape[0], self.args['num_labels']))
            for i in range(0, document_representations.shape[0], self.args['batch_size']):
                batch_document_tensors = document_representations[i:i + self.args['batch_size']].to(
                    device=self.args['device'])
                batch_document_sequence_lengths = document_sequence_lengths[i:i + self.args['batch_size']]

                prediction = self.bert_doc_classification(batch_document_tensors,
                                                          batch_document_sequence_lengths, device=self.args['device'])

                prediction = torch.sigmoid(prediction)
                predictions[i:i + self.args['batch_size']] = prediction

        if correct_output is None:
            return predictions.transpose(0, 1).cpu()
        else:
            for r in range(0, predictions.shape[0]):
                for c in range(0, predictions.shape[1]):
                    if predictions[r][c] > threshold:
                        predictions[r][c] = 1
                    else:
                        predictions[r][c] = 0
            predictions = predictions.transpose(0, 1)
            assert correct_output.shape == predictions.shape
            precisions = []
            recalls = []
            fmeasures = []

            for label_idx in range(predictions.shape[0]):
                correct = correct_output[label_idx].cpu().view(-1).numpy()
                predicted = predictions[label_idx].cpu().view(-1).numpy()
                present_f1_score = f1_score(correct, predicted, average='binary', pos_label=1)
                present_precision_score = precision_score(correct, predicted, average='binary', pos_label=1)
                present_recall_score = recall_score(correct, predicted, average='binary', pos_label=1)

                precisions.append(present_precision_score)
                recalls.append(present_recall_score)
                fmeasures.append(present_f1_score)

            micro_f1 = f1_score(correct_output.reshape(-1).numpy(), predictions.reshape(-1).numpy(), average='micro')

            print(micro_f1)
        self.bert_doc_classification.train()

    def save_checkpoint(self, checkpoint_path: str):
        """
        Saves an instance of the current model to the specified path.
        :return:
        """
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        self.log.info("Saving checkpoint: %s" % checkpoint_path)

        # save finetune parameters
        net = self.bert_doc_classification
        if isinstance(self.bert_doc_classification, nn.DataParallel):
            net = self.bert_doc_classification.module
        torch.save(net.state_dict(), os.path.join(checkpoint_path, 'model.bin'))
        # save configurations
        net.config.to_json_file(os.path.join(checkpoint_path, 'config.json'))
        # save exact vocabulary utilized
        self.bert_tokenizer.save_vocabulary(checkpoint_path)


class RobertaForDocumentClassification():
    def __init__(self, args=None,
                 labels=None,
                 device='cuda',
                 bert_model_path='vinai/phobert-base',
                 architecture="DocumentRobertaLSTM",
                 batch_size=10,
                 bert_batch_size=7,
                 learning_rate=5e-5,
                 weight_decay=0):
        if args is not None:
            self.args = vars(args)
        if not args:
            self.args = {}
            self.args['bert_model_path'] = bert_model_path
            self.args['device'] = device
            self.args['learning_rate'] = learning_rate
            self.args['weight_decay'] = weight_decay
            self.args['batch_size'] = batch_size
            self.args['num_labels'] = 1
            self.args['bert_batch_size'] = bert_batch_size
            self.args['architecture'] = architecture
            self.args['epochs'] = 10

        if 'fold' not in self.args:
            self.args['fold'] = 0

        self.log = logging.getLogger()
        # if bert_model_path.startswith('vinai'):
        self.bert_tokenizer = PhobertTokenizer.from_pretrained(self.args['bert_model_path'], do_lower_case=False)
        config = RobertaConfig.from_pretrained(self.args['bert_model_path'], num_labels=self.args['num_labels'],
                                               from_tf=False)
        # else:
        # self.bert_tokenizer = XLMRobertaTokenizer.from_pretrained(self.args['bert_model_path'], do_lower_case=False)
        # config = XLMRobertaConfig.from_pretrained(self.args['bert_model_path'], num_labels = self.args['num_labels'])

        self.bert_doc_classification = document_bert_architectures[self.args['architecture']].from_pretrained(
            self.args['bert_model_path'],
            config=config, args=self.args)
        self.bert_doc_classification.freeze_bert_encoder()
        self.bert_doc_classification.unfreeze_bert_encoder_last_layers()

        self.optimizer = torch.optim.Adam(
            self.bert_doc_classification.parameters(),
            weight_decay=self.args['weight_decay'],
            lr=self.args['learning_rate']
        )

    def fit(self, train, dev):
        """
        A list of
        :param documents: a list of documents
        :param labels: a list of label vectors
        :return:
        """

        train_documents, train_labels = train
        if dev is not None:
            dev_documents, dev_labels = dev

        document_representations, document_sequence_lengths = roberta_encode_documents(train_documents,
                                                                                       self.bert_tokenizer)

        correct_output = torch.FloatTensor(train_labels)

        loss_weight = ((correct_output.shape[0] / torch.sum(correct_output, dim=0)) - 1).to(device=self.args['device'])
        self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)

        assert document_representations.shape[0] == correct_output.shape[0]

        if torch.cuda.device_count() > 1:
            self.bert_doc_classification = torch.nn.DataParallel(self.bert_doc_classification)
        self.bert_doc_classification.to(device=self.args['device'])
        self.best_score = 0.0
        for epoch in range(1, self.args['epochs'] + 1):
            # shuffle
            self.bert_doc_classification.train()
            permutation = torch.randperm(document_representations.shape[0])
            document_representations = document_representations[permutation]
            document_sequence_lengths = document_sequence_lengths[permutation]
            correct_output = correct_output[permutation]
            print("Epoch %d..." % epoch)
            self.epoch = epoch
            epoch_loss = 0.0
            for i in range(0, document_representations.shape[0], self.args['batch_size']):
                if i / self.args['batch_size'] == 2:
                    print("Process %d batches" % i)

                batch_document_tensors = document_representations[i:i + self.args['batch_size']].to(
                    device=self.args['device'])
                batch_document_sequence_lengths = document_sequence_lengths[i:i + self.args['batch_size']]

                batch_predictions = self.bert_doc_classification(batch_document_tensors,
                                                                 batch_document_sequence_lengths,
                                                                 device=self.args['device'])

                batch_correct_output = correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])
                loss = self.loss_function(batch_predictions, batch_correct_output.unsqueeze(-1))
                epoch_loss += float(loss.item())

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss /= int(
                document_representations.shape[0] / self.args['batch_size'])  # divide by number of batches per epoch

            self.log.info('Epoch %i Completed: %f' % (epoch, epoch_loss))

            if dev is not None:
                self.predict((dev_documents, dev_labels))
                # self.save_checkpoint("trained_models/vbert")

    def predict(self, data, threshold=0.5):
        """
        A tuple containing
        :param data:
        :return:
        """
        document_representations = None
        document_sequence_lengths = None
        correct_output = None
        if isinstance(data, list):
            document_representations, document_sequence_lengths = roberta_encode_documents(data, self.bert_tokenizer)
        if isinstance(data, tuple) and len(data) == 2:
            self.log.info('Evaluating on Epoch %i' % (self.epoch))
            document_representations, document_sequence_lengths = roberta_encode_documents(data[0], self.bert_tokenizer)
            correct_output = torch.FloatTensor(data[1]).unsqueeze(-1).transpose(0, 1)

        self.bert_doc_classification.to(device=self.args['device'])
        self.bert_doc_classification.eval()
        with torch.no_grad():
            predictions = torch.empty((document_representations.shape[0], self.args['num_labels']))
            for i in range(0, document_representations.shape[0], self.args['batch_size']):
                batch_document_tensors = document_representations[i:i + self.args['batch_size']].to(
                    device=self.args['device'])
                batch_document_sequence_lengths = document_sequence_lengths[i:i + self.args['batch_size']]

                prediction = self.bert_doc_classification(batch_document_tensors,
                                                          batch_document_sequence_lengths, device=self.args['device'])

                prediction = torch.sigmoid(prediction)
                predictions[i:i + self.args['batch_size']] = prediction

        if correct_output is None:
            return predictions.transpose(0, 1).cpu()
        else:
            from sklearn.metrics import roc_auc_score
            roc_score = roc_auc_score(correct_output.reshape(-1).numpy(),
                                      predictions.transpose(0, 1).cpu().reshape(-1).numpy())
            if self.best_score < roc_score:
                self.best_score = roc_score
                self.save_checkpoint("trained_models/phobert_base")

            for r in range(0, predictions.shape[0]):
                for c in range(0, predictions.shape[1]):
                    if predictions[r][c] > threshold:
                        predictions[r][c] = 1
                    else:
                        predictions[r][c] = 0
            predictions = predictions.transpose(0, 1)
            assert correct_output.shape == predictions.shape
            precisions = []
            recalls = []
            fmeasures = []

            for label_idx in range(predictions.shape[0]):
                correct = correct_output[label_idx].cpu().view(-1).numpy()
                predicted = predictions[label_idx].cpu().view(-1).numpy()
                present_f1_score = f1_score(correct, predicted, average='binary', pos_label=1)
                present_precision_score = precision_score(correct, predicted, average='binary', pos_label=1)
                present_recall_score = recall_score(correct, predicted, average='binary', pos_label=1)

                precisions.append(present_precision_score)
                recalls.append(present_recall_score)
                fmeasures.append(present_f1_score)

            micro_f1 = f1_score(correct_output.reshape(-1).numpy(), predictions.reshape(-1).numpy(), average='micro')

            print(micro_f1, roc_score)
        self.bert_doc_classification.train()

    def save_checkpoint(self, checkpoint_path: str):
        """
        Saves an instance of the current model to the specified path.
        :return:
        """
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        self.log.info("Saving checkpoint: %s" % checkpoint_path)

        # save finetune parameters
        net = self.bert_doc_classification
        if isinstance(self.bert_doc_classification, nn.DataParallel):
            net = self.bert_doc_classification.module
        torch.save(net.state_dict(), os.path.join(checkpoint_path, 'pytorch_model.bin'))
        # save configurations
        net.config.to_json_file(os.path.join(checkpoint_path, 'config.json'))
        # save exact vocabulary utilized
        self.bert_tokenizer.save_vocabulary(checkpoint_path)
