
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import LongformerTokenizer, LongformerModel
import math
import random
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import Counter
from scipy.optimize import linear_sum_assignment



''' create train/dev/test file paths '''

dev_file_paths = []
dev_path = "./Propaganda/json_annotations/dev"
dev_files = os.listdir(dev_path)
for file_i in range(len(dev_files)):
    dev_file_paths.append(dev_path + "/" + dev_files[file_i])

test_file_paths = []
test_path = "./Propaganda/json_annotations/test"
test_files = os.listdir(test_path)
for file_i in range(len(test_files)):
    test_file_paths.append(test_path + "/" + test_files[file_i])

train_file_paths = []
train_path = "./Propaganda/json_annotations/train"
train_files = os.listdir(train_path)
for file_i in range(len(train_files)):
    train_file_paths.append(train_path + "/" + train_files[file_i])




''' hyper-parameter '''

MAX_LEN = 4096
num_epochs = 6
batch_size = 1
check_times = 10 * num_epochs

no_decay = ['bias', 'LayerNorm.weight']
longformer_weight_decay = 1e-2
non_longformer_weight_decay = 1e-2

warmup_proportion = 0.0
non_longformer_lr = 1e-4
longformer_lr = 1e-5





''' custom dataset '''

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

class custom_dataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with open(file_path, "r") as in_json:
            article_json = json.load(in_json)

        label_dict = {'NA': 0, 'Loaded_Language': 1, 'Name_Calling,Labeling': 2, 'Repetition': 3, 'Exaggeration,Minimisation': 4,
                      'Doubt': 5, 'Appeal_to_fear-prejudice': 6, 'Causal_Oversimplification': 7, 'Flag-Waving': 8,
                      'Appeal_to_Authority': 9, 'Slogans': 10, 'Black-and-White_Fallacy': 11, 'Red_Herring': 12,
                      'Reductio_ad_hitlerum': 13, 'Thought-terminating_Cliches': 14, 'Bandwagon': 15, 'Whataboutism': 16,
                      'Obfuscation,Intentional_Vagueness,Confusion': 17, 'Straw_Men': 18}

        label_token = []
        # label_token[i,:] = [start, end, token-level propaganda], where [start, end] is the corresponding index in input_ids of ith token start token <s>
        # input_ids[range(start, end)] can extract the corresponding ith token start token <s>


        news_discourse_prob = []
        # news_discourse_prob[i,:] = [news_discourse_prob_0, news_discourse_prob_1, ..., news_discourse_prob_8]

        pdtb_relation_prob = []
        # pdtb_relation_prob[i,:] = [discourse_relation_prob_comp, ..., discourse_relation_prob_expan] between ith and (i+1)th sentence


        input_ids = []
        attention_mask = []

        for sent_i in range(len(article_json['sentences'])):
            sentence_text = article_json['sentences'][sent_i]['sentence_text']

            start = len(input_ids)
            sentence_encoding = tokenizer.encode_plus('<s>' + sentence_text + '</s>', add_special_tokens=False)

            if start + len(sentence_encoding['input_ids']) < MAX_LEN:  # truncate to MAX_LEN
                input_ids.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['input_ids'])  # the <s> start token
                attention_mask.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['attention_mask'])

                for token_i in range(len(article_json['sentences'][sent_i]['tokens_dicts_list'])):
                    token_text = article_json['sentences'][sent_i]['tokens_dicts_list'][token_i]['token']
                    token_label = label_dict[article_json['sentences'][sent_i]['tokens_dicts_list'][token_i]['token_label']]
                    if token_label != 0:
                        token_label = 1

                    start = len(input_ids)
                    input_ids.extend(tokenizer.encode_plus(' ' + token_text, add_special_tokens=False)['input_ids'])  # the <s> start token
                    attention_mask.extend(tokenizer.encode_plus(' ' + token_text, add_special_tokens=False)['attention_mask'])
                    end = len(input_ids)
                    label_token.append([start, end, token_label])

                    news_discourse_prob.append([article_json['sentences'][sent_i]['news_discourse_role_prob_0'],
                                                article_json['sentences'][sent_i]['news_discourse_role_prob_1'],
                                                article_json['sentences'][sent_i]['news_discourse_role_prob_2'],
                                                article_json['sentences'][sent_i]['news_discourse_role_prob_3'],
                                                article_json['sentences'][sent_i]['news_discourse_role_prob_4'],
                                                article_json['sentences'][sent_i]['news_discourse_role_prob_5'],
                                                article_json['sentences'][sent_i]['news_discourse_role_prob_6'],
                                                article_json['sentences'][sent_i]['news_discourse_role_prob_7'],
                                                article_json['sentences'][sent_i]['news_discourse_role_prob_8']])

                    if article_json['sentences'][sent_i]['latter_discourse_relation'] != -1:
                        pdtb_relation_prob.append([article_json['sentences'][sent_i]['latter_discourse_relation_prob_comp'],
                                                   article_json['sentences'][sent_i]['latter_discourse_relation_prob_cont'],
                                                   article_json['sentences'][sent_i]['latter_discourse_relation_prob_temp'],
                                                   article_json['sentences'][sent_i]['latter_discourse_relation_prob_expan']])
                    else:
                        pdtb_relation_prob.append([0, 0, 0, 0])

                input_ids.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['input_ids'])  # the </s> end token
                attention_mask.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['attention_mask'])

            else:
                break

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        label_token = torch.tensor(label_token)
        news_discourse_prob = torch.tensor(news_discourse_prob)
        pdtb_relation_prob = torch.tensor(pdtb_relation_prob)


        dict = {"input_ids": input_ids, "attention_mask": attention_mask, "label_token": label_token,
                "news_discourse_prob": news_discourse_prob, "pdtb_relation_prob": pdtb_relation_prob}

        return dict





''' model '''

class Token_Embedding(nn.Module):

    # input: input_ids, attention_mask, 1 article * number of tokens
    # output: number of tokens * 768, dealing with one article at one time

    def __init__(self):
        super(Token_Embedding, self).__init__()

        self.longformermodel = LongformerModel.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True, )

    def forward(self, input_ids, attention_mask):

        outputs = self.longformermodel(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[2]
        token_embeddings_layers = torch.stack(hidden_states, dim = 0)  # 13 layer * batch_size (1) * number of tokens * 768
        token_embeddings_layers = token_embeddings_layers[:, 0, :, :] # 13 layer * number of tokens * 768
        token_embeddings = torch.sum(token_embeddings_layers[-4:, :, :], dim = 0) # sum up the last four layers, number of tokens * 768

        return token_embeddings



class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.token_embedding = Token_Embedding()

        self.bilstm_token = nn.LSTM(input_size = 768, hidden_size = 384, batch_first=True, bidirectional=True)
        self.bilstm_sent = nn.LSTM(input_size = 768, hidden_size = 384, batch_first=True, bidirectional=True)

        self.propaganda_1 = nn.Linear(768 + 9 + 4, 768, bias=True)
        nn.init.xavier_uniform_(self.propaganda_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.propaganda_1.bias)

        self.propaganda_2 = nn.Linear(768, 2, bias=True)
        nn.init.xavier_uniform_(self.propaganda_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.propaganda_2.bias)

        self.relu = nn.ReLU()
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='mean')


    def forward(self, input_ids, attention_mask, label_token, news_discourse_prob, pdtb_relation_prob):

        token_embeddings = self.token_embedding(input_ids, attention_mask) # number of tokens * 768

        # token-level bi-lstm layer
        token_embeddings = token_embeddings.view(1, token_embeddings.shape[0], token_embeddings.shape[1])

        h0_token = torch.zeros(2, 1, 384).cuda().requires_grad_()
        c0_token = torch.zeros(2, 1, 384).cuda().requires_grad_()

        token_embeddings, (_, _) = self.bilstm_token(token_embeddings, (h0_token, c0_token)) # batch_size 1 * number of tokens * 768
        token_embeddings = token_embeddings[0, :, :] # number of tokens * 768

        # target token embedding
        target_token_embedding = token_embeddings[label_token[:, 0]] # nrow of label_token * 768

        # additional news discourse feature
        target_token_embedding = torch.cat((target_token_embedding, news_discourse_prob), dim = 1)

        # additional pdtb relation feature
        target_token_embedding = torch.cat((target_token_embedding, pdtb_relation_prob), dim = 1)

        # token-level propaganda task
        propaganda_scores = self.propaganda_2(self.relu(self.propaganda_1(target_token_embedding)))
        propaganda_loss = self.crossentropyloss(propaganda_scores, label_token[:, 2])

        return propaganda_scores, propaganda_loss





''' evaluate '''

def evaluate(model, eval_dataloader, verbose):

    model.eval()

    for step, batch in enumerate(eval_dataloader):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label_token = batch['label_token'][0]
        news_discourse_prob = batch['news_discourse_prob'][0]
        pdtb_relation_prob = batch['pdtb_relation_prob'][0]

        input_ids, attention_mask, label_token, news_discourse_prob, pdtb_relation_prob = \
            input_ids.to(device), attention_mask.to(device), label_token.to(device), news_discourse_prob.to(device), pdtb_relation_prob.to(device)

        with torch.no_grad():
            propaganda_scores, propaganda_loss = model(input_ids, attention_mask, label_token, news_discourse_prob, pdtb_relation_prob)

        decision = torch.argmax(propaganda_scores, dim = 1).view(propaganda_scores.shape[0], 1)
        true_label = label_token[:, 2].view(propaganda_scores.shape[0], 1)

        if step == 0:
            decision_onetest = decision
            true_label_onetest = true_label
        else:
            decision_onetest = torch.cat((decision_onetest, decision), dim=0)
            true_label_onetest = torch.cat((true_label_onetest, true_label), dim=0)

    decision_onetest = decision_onetest.to('cpu').numpy()
    true_label_onetest = true_label_onetest.to('cpu').numpy()

    if verbose:
        print("Macro: ", precision_recall_fscore_support(true_label_onetest, decision_onetest, average='macro'))
        print("Positive: ", precision_recall_fscore_support(true_label_onetest, decision_onetest, average='binary'))

    macro_F = precision_recall_fscore_support(true_label_onetest, decision_onetest, average='macro')[2]
    positive_F = precision_recall_fscore_support(true_label_onetest, decision_onetest, average='binary')[2]

    return macro_F, positive_F






''' train '''



import time
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()





seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

model = Model()
model.cuda()
param_all = list(model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('longformer' in n))], 'lr': longformer_lr, 'weight_decay': longformer_weight_decay},
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and (not 'longformer' in n))], 'lr': non_longformer_lr, 'weight_decay': non_longformer_weight_decay},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('longformer' in n))], 'lr': longformer_lr, 'weight_decay': 0.0},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and (not 'longformer' in n))], 'lr': non_longformer_lr, 'weight_decay': 0.0}]
# optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8)

train_dataset = custom_dataset(train_file_paths)
dev_dataset = custom_dataset(dev_file_paths)
test_dataset = custom_dataset(test_file_paths)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_train_steps = num_epochs * len(train_dataloader)
warmup_steps = int(warmup_proportion * num_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)


best_dev_macro_F = 0
best_dev_positive_F = 0


for epoch_i in range(num_epochs):

    np.random.shuffle(train_file_paths)  # shuffle training data
    train_dataset = custom_dataset(train_file_paths)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i, num_epochs))
    print('Training...')

    t0 = time.time()
    total_propaganda_loss = 0
    num_batch = 0

    for step, batch in enumerate(train_dataloader):

        if step % ((len(train_dataloader) * num_epochs) // check_times) == 0:

            elapsed = format_time(time.time() - t0)
            if num_batch != 0:
                avg_propaganda_loss = total_propaganda_loss / num_batch
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    propaganda loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_propaganda_loss))
            else:
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            total_propaganda_loss = 0
            num_batch = 0

            # evaluate on dev set

            macro_F, positive_F = evaluate(model, dev_dataloader, verbose = 0)
            if macro_F > best_dev_macro_F:
                torch.save(model.state_dict(), "./saved_models/token_identification_additional_best_dev_macro_F.ckpt")
                best_dev_macro_F = macro_F
            if positive_F > best_dev_positive_F:
                torch.save(model.state_dict(), "./saved_models/token_identification_additional_best_dev_positive_F.ckpt")
                best_dev_positive_F = positive_F



        # train

        model.train()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label_token = batch['label_token'][0]
        news_discourse_prob = batch['news_discourse_prob'][0]
        pdtb_relation_prob = batch['pdtb_relation_prob'][0]

        input_ids, attention_mask, label_token, news_discourse_prob, pdtb_relation_prob = \
            input_ids.to(device), attention_mask.to(device), label_token.to(device), news_discourse_prob.to(device), pdtb_relation_prob.to(device)

        optimizer.zero_grad()

        propaganda_scores, propaganda_loss = \
            model(input_ids, attention_mask, label_token, news_discourse_prob, pdtb_relation_prob)

        total_propaganda_loss += propaganda_loss.item()
        num_batch += 1

        propaganda_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()



    elapsed = format_time(time.time() - t0)
    if num_batch != 0:
        avg_propaganda_loss = total_propaganda_loss / num_batch
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    propaganda loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_propaganda_loss))
    else:
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

    total_propaganda_loss = 0
    num_batch = 0

    # evaluate on dev set

    macro_F, positive_F = evaluate(model, dev_dataloader, verbose = 0)
    if macro_F > best_dev_macro_F:
        torch.save(model.state_dict(), "./saved_models/token_identification_additional_best_dev_macro_F.ckpt")
        best_dev_macro_F = macro_F
    if positive_F > best_dev_positive_F:
        torch.save(model.state_dict(), "./saved_models/token_identification_additional_best_dev_positive_F.ckpt")
        best_dev_positive_F = positive_F




print("Evaluation on Test set")


# best dev macro F on test set

print("Best dev macro F is: {:}".format(best_dev_macro_F))
model.load_state_dict(torch.load("./saved_models/token_identification_additional_best_dev_macro_F.ckpt", map_location=device))
model.eval()
macro_F, positive_F = evaluate(model, test_dataloader, verbose = 1)


# best dev positive F on test set

print("Best dev positive F is: {:}".format(best_dev_positive_F))
model.load_state_dict(torch.load("./saved_models/token_identification_additional_best_dev_positive_F.ckpt", map_location=device))
model.eval()
macro_F, positive_F = evaluate(model, test_dataloader, verbose = 1)
















# stop here
