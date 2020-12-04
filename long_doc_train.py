from long_document_model import *
from preprocess import *
from utils import *

if __name__ == '__main__':
    train_df = pd.read_csv('public_train.csv')
    test_df = pd.read_csv('public_test.csv')
    val_df = pd.read_csv('val.csv')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(torch.cuda.get_device_name())
    else:
        device = torch.device('cpu')

    seed_everything(69)

    vncorenlp = VnCoreNLP('VnCoreNLP-1.1.1.jar', annotators='wseg')
    tweetTokenizer = TweetTokenizer()

    # process training set
    error_label_idx = []
    tr_texts = []
    for i, post in enumerate(train_df.post_message):
        if not isnan(post):
            tr_texts.append(normalizePost(post, use_segment=True, remove_punc_stopword=False))
        else:
            error_label_idx.append(i)
    tr_labels = train_df.iloc[~train_df.index.isin(error_label_idx)].label.to_list()

    # process validation set
    error_label_idx = []
    vl_texts = []
    for i, post in enumerate(val_df.post_message):
        if not isnan(post):
            vl_texts.append(normalizePost(post, use_segment=True, remove_punc_stopword=False))
        else:
            error_label_idx.append(i)
    vl_labels = val_df.iloc[~val_df.index.isin(error_label_idx)].label.to_list()

    document_model = RobertaForDocumentClassification()
    document_model.fit((tr_texts, tr_labels), (vl_texts, vl_labels))
