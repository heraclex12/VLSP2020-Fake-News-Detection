from eval import *


EPOCHS = 20
BATCH_SIZE = 8
ACCUMULATION_STEPS = 5


if __name__ == '__main__':
    train_df = pd.read_csv('public_train.csv')
    val_df = pd.read_csv('val.csv')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(torch.cuda.get_device_name())
    else:
        device = torch.device('cpu')

    # load config
    config_path = 'config/electra_1.json'
    single_model_config = json.load(open(config_path, 'r'))

    # init external tools
    vncorenlp = VnCoreNLP('VnCoreNLP-1.1.1.jar', annotators='wseg')
    tweet_tokenizer = TweetTokenizer()

    # process training set
    error_label_idx = []
    tr_texts = []
    for i, post in enumerate(train_df.post_message):
        if not isnan(post):
            tr_texts.append(normalizePost(post, tweet_tokenizer, vncorenlp, use_segment=single_model_config['use_wordsegment'],
                                          remove_punc_stopword=single_model_config['remove_punc_stopword']))
        else:
            error_label_idx.append(i)
    tr_labels = train_df.iloc[~train_df.index.isin(error_label_idx)].label.to_list()
    train_ids, train_masks, train_labels = convert_samples_to_ids(tr_texts, tr_labels)
    train_dataset = torch.utils.data.TensorDataset(train_ids, train_masks, train_labels)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)

    # process validation set
    error_label_idx = []
    vl_texts = []
    for i, post in enumerate(val_df.post_message):
        if not isnan(post):
            vl_texts.append(normalizePost(post, tweet_tokenizer, vncorenlp, use_segment=single_model_config['use_wordsegment'],
                                          remove_punc_stopword=single_model_config['remove_punc_stopword']))
        else:
            error_label_idx.append(i)
    vl_labels = val_df.iloc[~val_df.index.isin(error_label_idx)].label.to_list()
    val_ids, val_masks, val_labels = convert_samples_to_ids(vl_texts, vl_labels)
    val_dataset = torch.utils.data.TensorDataset(val_ids, val_masks, val_labels)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    for _ in range(10):
        seed = np.random.randint(0, 10000)
        seed_everything(seed)
        # init tokenizer
        if single_model_config['model_type'] == 'BERT':
            print("===Use BERT model===")
            tokenizer = BertTokenizer.from_pretrained(single_model_config['model_name'], do_lower_case=False)
            tokenizer.add_tokens(['<url>'])
            config = BertConfig.from_pretrained(single_model_config['model_name'], num_labels=1,
                                                output_hidden_states=True)
            model = BertReINTELClassification.from_pretrained(single_model_config['model_name'], config=config)
            model.to(device)
            tsfm = model.bert
        elif single_model_config['model_type'] == 'ROBERTA':
            print("===Use ROBERTA model===")
            tokenizer = PhobertTokenizer.from_pretrained(single_model_config['model_name'])
            tokenizer.add_tokens(['<url>'])
            config = RobertaConfig.from_pretrained(single_model_config['model_name'], num_labels=1,
                                                   output_hidden_states=True)
            model = RobertaReINTELClassification.from_pretrained(single_model_config['model_name'], config=config)
            model.resize_token_embeddings(len(tokenizer))
            model.to(device)
            tsfm = model.roberta
        elif single_model_config['model_type'] == 'ELECTRA':
            print("===Use ELECTRA model===")
            tokenizer = ElectraTokenizer.from_pretrained(single_model_config['model_name'], do_lower_case=False)
            tokenizer.add_tokens(['<url>'])
            config = ElectraConfig.from_pretrained(single_model_config['model_name'], num_labels=1,
                                                   output_hidden_states=True, output_attentions=False)
            model = ElectraReINTELClassification.from_pretrained(single_model_config['model_name'], config=config)
            model.resize_token_embeddings(len(tokenizer))
            model.to(device)
            tsfm = model.electra
        elif single_model_config['model_type'] == 'XML_ROBERTA':
            print("===Use XML-ROBERTA model===")
            tokenizer = XLMRobertaTokenizer.from_pretrained(single_model_config['model_name'], do_lower_case=False)
            tokenizer.add_tokens(['<url>'])
        else:
            print("Model type invalid!!!")

        num_train_optimization_steps = int(EPOCHS * len(train_dataset) / BATCH_SIZE / ACCUMULATION_STEPS)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(np in n for np in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(np in n for np in no_decay)], 'weight_decay': 0.01}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                    num_training_steps=num_train_optimization_steps)
        scheduler0 = get_constant_schedule(optimizer)

        # freeze head layers
        for child in tsfm.children():
            for param in child.parameters():
                param.requires_grad = False

        # Convert to iterator

        frozen = True
        best_score = 0.0

        for epoch in range(EPOCHS + 1):
            # unfreeze
            if epoch > 0 and frozen:
                for child in tsfm.children():
                    for param in child.parameters():
                        param.requires_grad = True

                frozen = False
                del scheduler0
                torch.cuda.empty_cache()

            print('\n------ Start training on Epoch: %d/%d' % (epoch, EPOCHS))

            avg_loss = 0
            avg_accuracy = 0
            # Training process
            model.train()
            for i, (input_ids, attention_mask, y_batch) in enumerate(train_loader):
                if (i % 20 == 0 and not i == 0) or (i == len(train_loader)):
                    print(f'Batch {i} of {len(train_loader)}...')

                optimizer.zero_grad()
                y_pred = model(input_ids.to(device), attention_mask=attention_mask.to(device))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred.view(-1).to(device),
                                                                            y_batch.float().to(device))
                loss = loss.mean()
                loss.backward()
                optimizer.step()

                lossf = loss.item()
                avg_loss += loss.item() / len(train_loader)

            if not frozen:
                scheduler.step()
            else:
                scheduler0.step()
            optimizer.zero_grad()
            # save_checkpoint(model, tokenizer, 'trained_models/bert_multilingual', epoch=epoch)

            roc_score = eval(val_loader, model, epoch, seed)
            if roc_score >= best_score:
                save_checkpoint(model, tokenizer, 'trained_models/phobert_random', epoch=seed)
                best_score = roc_score
                print("Updated best score model!!! -------<{}>" % best_score)
            print('==========================================')
