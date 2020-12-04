from preprocess import *
from utils import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import csv
import pandas as pd
from models import *
import json
from transformers import *
from nltk.tokenize import TweetTokenizer
from vncorenlp import VnCoreNLP


def eval(val_loader, model, epoch, device):
    # Evaluate model
    model.eval()
    y_val = []
    val_preds = None
    print(f"EPOCH {epoch}: ===EVALUATION===")
    for (input_ids, attention_mask, y_batch) in val_loader:
        y_pred = model(input_ids.to(device), attention_mask=attention_mask.to(device))
        y_pred = y_pred.squeeze().detach().cpu().numpy()
        val_preds = np.atleast_1d(y_pred) if val_preds is None else np.concatenate(
            [val_preds, np.atleast_1d(y_pred)])
        y_val.extend(y_batch.tolist())

    val_preds = sigmoid(val_preds)
    score = f1_score(y_val, val_preds > 0.5, pos_label=0)
    roc_score = roc_auc_score(y_val, val_preds)
    print(f"PREDICT {sum(val_preds <= 0.5)} INFORMATIVES")
    print(f"ACTUALY {len(y_val) - sum(y_val)} INFORMATIVES")

    print(f"\n----- F1 score @0.5 = {score:.4f}\nROC-AUC Score = {roc_score:.4f}")
    return roc_score


def predict(test_df, model, config, tweet_tokenizer, vncorenlp):
    test_normalized_texts = []
    test_post_ids = []
    for row in test_df.iterrows():
        if not isnan(row[1]['post_message']):
            test_normalized_texts.append(
                normalizePost(row[1]['post_message'], tweet_tokenizer, vncorenlp, use_segment=config['use_wordsegment'],
                              remove_punc_stopword=config['remove_punc_stopword']))
            test_post_ids.append(row[1]['id'])

    test_ids, test_masks = convert_samples_to_ids(test_normalized_texts)

    test_dataset = torch.utils.data.TensorDataset(test_ids, test_masks)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    model.eval()
    test_preds = None
    for i, (input_ids, masks) in enumerate(test_dataloader):
        if i % 20 == 0 or i == len(test_dataloader):
            print(f"Predicted {i} posts.")
        y_pred = model(input_ids.cuda(), attention_mask=masks.cuda())
        y_pred = y_pred.squeeze().detach().cpu().numpy()
        test_preds = np.atleast_1d(y_pred) if test_preds is None else np.concatenate(
            [test_preds, np.atleast_1d(y_pred)])

    test_preds = sigmoid(test_preds)
    test_preds = test_preds.tolist()

    with open('results.csv', 'w') as out:
        writer = csv.writer(out)
        for post_id, test_pred in zip(test_post_ids, test_preds):
            writer.writerow([post_id, test_pred])


if __name__ == '__main__':
    test_df = pd.read_csv('final_private_test_dropped_no_label.csv')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(torch.cuda.get_device_name())
    else:
        device = torch.device('cpu')

    config = ElectraConfig.from_pretrained('trained_models/electra_512', num_labels=1, output_hidden_states=True)
    model = ElectraReINTELClassification.from_pretrained('trained_models/electra_512_tmp/model_best.bin', config=config)
    model.to(device)
    tokenizer = ElectraTokenizer.from_pretrained('trained_models/electra_512', do_lower_case=False)

    config_path = 'config/electra_1.json'
    single_model_config = json.load(open(config_path, 'r'))

    vncorenlp = VnCoreNLP('VnCoreNLP-1.1.1.jar', annotators='wseg')
    tweet_tokenizer = TweetTokenizer()

    predict(test_df, model, single_model_config, tweet_tokenizer, vncorenlp)