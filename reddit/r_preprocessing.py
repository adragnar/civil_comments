#Modify existing reddit datasets with a toxicity column
import argparse
import os
from os.path import join, dirname

import sys

sys.path.append(join(os.getcwd(), 'launchfiles'))
sys.path.append(os.getcwd())

import copy
import pickle
import socket

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
STOPWORDS = set(stopwords.words('english'))

from tqdm import tqdm

import data_proc
import models
import setup_params as setup
import r_setup_params as reddit_setup

#For BERT
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import torch.utils.data
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import os
# from os.path import join
# import warnings
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)




def preprocess_data(data, rel_cols, tox_thresh=None, c_len=15, \
                                 text_clean='reg', stopwords=[]):
    '''Raw preprocessing to be done before data is used on both data All non nan
       values stuff controlled by kwargs
    :param data: pandas dataframe.
    :param rel_cols dictionary mapping keys (data, labels) to names in given dset
    :param tox_thresh: threshold for binarizing tox label (if dataset has labels)
    :param c_len: min length of comment
    :param social_media: Whether to do a variety of social-media specific preprocessing
    :param stopwords: words to take out if social_media=True'''

    #Clean Target
    #remove Nans in comment text column
    data['test_nan']= data[rel_cols['data']].apply(lambda x: 1 if type(x) == str else 0)
    data = data[data['test_nan'] == 1]; data.reset_index(inplace=True, drop=True)
    data.drop(['test_nan'], axis=1, inplace=True)

    #Do social media preprocess
    if text_clean == 'sm':
        def proc_social(words, stopwords=[]):
            '''Remove html tags and non-words from text preproc
               words: list of tokens'''
            #html
            words = [re.sub(r'<.*?>', '', w).lower() for w in words]

            #stopwords and alphanumeric
            words = [re.sub(r'\W+', '', w).lower() for w in words \
                     if re.sub(r'\W+', '', w).lower() not in stopwords]

            #Remove letter artefacts
            words = [w for w in words if ((len(w) > 1) or (w in ['a', 'i', 'u']))]
            return words

        text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user', \
                       'time', 'date', 'number'],
            fix_html=True,  # fix HTML tokens
            segmenter="english",
            corrector="english",
            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=False,  # spell correction for elongated words
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
        )

        data[rel_cols['data']] = data[rel_cols['data']].apply( \
            lambda s: " ".join(proc_social(text_processor.pre_process_doc(s), \
                                            stopwords=STOPWORDS)))
    elif text_clean == 'reg':  #If no social media
        def proc(txt, stopwords=[]):
            words = txt.split(" ")
            words = [re.sub(r'\W+', '', w).lower() for w in words \
                     if re.sub(r'\W+', '', w).lower() not in stopwords]
            return words

        data[rel_cols['data']] = data[rel_cols['data']].apply(\
                        lambda s: " ".join(proc(s, stopwords=STOPWORDS)))

    elif text_clean == 'na':
        pass
    else:
        raise Exception('Unimplemented cleaning')

    #Remove too small comments
    if c_len is not None:
        data['test_len'] = data[rel_cols['data']].apply(lambda x: 1 if (len(str(x)) > c_len) else 0)
        data = data[data['test_len'] == 1]; data.reset_index(inplace=True, drop=True)
        data.drop(['test_len'], axis=1, inplace=True)

    #Binarize labels if labelled
    if tox_thresh is not None:
        data[rel_cols['labels']] = data[rel_cols['labels']].apply((lambda x: 1 if x > tox_thresh else 0))

    return data

def bert_add_toxlabel(old_fpaths, new_fpaths, rel_cols, \
                                 BERT_MODEL_PATH, BERT_FINETUNE_PATH):
    def convert_lines(example, max_seq_length,tokenizer):
        max_seq_length -=2
        all_tokens = []
        longer = 0
        for text in tqdm(example):
            tokens_a = tokenizer.tokenize(text)
            if len(tokens_a)>max_seq_length:
                tokens_a = tokens_a[:max_seq_length]
                longer += 1
            one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
            all_tokens.append(one_token)
        return np.array(all_tokens)

    #Set up the BERT model
    MAX_SEQUENCE_LENGTH = 400
    SEED = 1234

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    bert_config = BertConfig(join(BERT_FINETUNE_PATH, 'bert_config.json'))
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, \
                                      cache_dir=None,do_lower_case=True)

    model = BertForSequenceClassification(bert_config, num_labels=1)
    model.load_state_dict(torch.load(join(BERT_FINETUNE_PATH, \
                                  "bert_pytorch.bin"), map_location=device))
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    for fname, new_fname in zip(old_fpaths, new_fpaths):
        data = pd.read_csv(fname)
        data['proc_body'] = data[rel_cols['data']]
        data = preprocess_data(data, {'data':'proc_body'}, c_len=15, text_clean='na')
        X = convert_lines(data['proc_body'].fillna("DUMMY_VALUE"), \
                                    MAX_SEQUENCE_LENGTH, tokenizer)
        bsize = 32
        test_preds = np.zeros((len(X)))
        test = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.long))
        test_loader = torch.utils.data.DataLoader(test, batch_size=bsize, shuffle=False)
        tk0 = tqdm(test_loader)
        for i, (x_batch,) in tqdm(enumerate(tk0)):
            pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
            test_preds[i * bsize:(i + 1) * bsize] = pred[:, 0].detach().cpu().squeeze().numpy()

        test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()

        #Save Results
        data['pred_toxicity'] = test_pred
        data.to_csv(new_fname)


def add_toxlabel(old_fpaths, new_fpaths, rel_cols, mpath, e_path):
    ''':param fname: path to reddit dataset
       :param new_fname: path to labelled reddit dataset
       :param mpath: path to model file
       :param epath: path to embeddings file'''

    t = data_proc.get_word_transform('embed', e_path, proc=False)
    print('WE loaded')
    for fname, new_fname in zip(old_fpaths, new_fpaths):
        data = pd.read_csv(fname)
        data['proc_body'] = data[rel_cols['data']]

        data = preprocess_data(data, {'data':'proc_body'}, c_len=0, text_clean='reg')
        data_embed = t(data['proc_body'])
        print('data')
        model = data_proc.load_saved_model(mpath)
        print('model')
        data['pred_toxicity'] = model.predict(data_embed)
        data.to_csv(new_fname)


if __name__ == '__main__':
    assert (os.getcwd().split('/')[-1] == 'civil_comments') or \
                    (os.getcwd().split('/')[-1] == 'civil_liberties')

    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument("old_fpaths", type=str, default=None)
    parser.add_argument("new_fpaths", type=str, default=None)
    parser.add_argument("model", type=str, default=None)
    parser.add_argument("data_type", type=str, default=None)
    args = parser.parse_args()

    old_fpaths = [args.old_fpaths]  #'reddit/data/orig/2014_gendered.csv']
    new_fpaths = [args.new_fpaths]  #'reddit/data/labeled/2014_gendered_bert_labeled.csv']

    if args.data_type == 'jigsaw':
        rel_cols = {'data':'comment_text'}
    elif args.data_type == 'reddit':
        rel_cols = {'data':'body'}

    if args.model == 'bert':
        pretrained_path = 'reddit/labelgen_models/bert/bert_pretrained/uncased_L-12_H-768_A-12/'
        finetuned_path = 'reddit/labelgen_models/bert/bert_finetuned/'
        bert_add_toxlabel(old_fpaths, \
                          new_fpaths, \
                          rel_cols, \
                          pretrained_path, \
                          finetuned_path)

    elif args.model == 'mlp':
        m_path = 'reddit/labelgen_models/0810_labelgen.pkl'
        add_toxlabel(old_fpaths, \
                     new_fpaths, \
                     rel_cols, \
                     m_path, \
                     setup.get_wordvecspath())
