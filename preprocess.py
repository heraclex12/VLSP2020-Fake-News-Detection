import emoji
import re
import torch


def normalizeToken(token):
    if len(token) == 1:
        return emoji.demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def isnan(s):
    return s != s


def normalizePost(post, tweet_tokenizer, vncorenlp, use_segment=False, remove_punc_stopword=False):
    tokens = tweet_tokenizer.tokenize(post.replace("’", "'").replace("…", "..."))
    post = " ".join(tokens)
    if use_segment:
        tokens = vncorenlp.tokenize(post.replace("’", "'").replace("…", "..."))
        tokens = [t for ts in tokens for t in ts]
    normPost = " ".join(tokens)
    # normPost = " ".join([normalizeToken(token) for token in tokens if not token.startswith('#') and token not in emoji.UNICODE_EMOJI])
    # normPost = " ".join([normalizeToken(token) for token in tokens if token not in emoji.UNICODE_EMOJI])

    normPost = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normPost)
    normPost = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normPost)
    normPost = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normPost)
    if use_segment:
        normPost = normPost.replace('< url >', '<url>')
        normPost = re.sub(r"# (\w+)", r'#\1', normPost)

    # normPost = ' '.join(normPost.split())
    return normPost


def convert_samples_to_ids(texts, tokenizer, max_seq_length=256, labels=None):
    input_ids, attention_masks = [], []
    for text in texts:
        inputs = tokenizer.encode_plus(text, padding='max_length', max_length=max_seq_length, truncation=True)
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])

    if labels is not None:
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long)


def get_max_seq(texts, tokenizer):
    max_seq_length = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        # max_seq_length = max(max_seq_length, len(tokens))
        max_seq_length.append(len(tokens))

    return max_seq_length
