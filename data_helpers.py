import json

import numpy as np
import re
import itertools
from collections import Counter
import os

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def create_one_hot_encoding(dialog_act, labels_dict):
    if len(labels_dict.keys()) == 0:
        da2index = {
            "statement":0,
            "directive":1,
            "commissive":2,
            "infoq":3,
            "checkq":4,
            "choiceq":5,
            "answer":6,
            "feedback":7,
            "thanking":8,
            "apology":9,
            "goodbye":10,
            "greeting":11
        }
    else:
        da2index = labels_dict
    vector = np.zeros(len(da2index.keys()))
    index = None
    if dialog_act is not None:
        index = da2index.get(dialog_act.lower(), None)
    if index is not None:
        vector[index] = 1
    return vector

def append_to_additional_file(message):
    with open("/home/giuliano.tortoreto/slu/logging_eval_info.txt",'a') as out_file:
        out_file.write(str(message)+'\n')

def sample2text_prev_da(examples, labels_dict ={}, n_prev_da=1):
    texts = []
    das = []
    prev_das = []
    for e in examples :
        if e.strip() == "":
            continue
        text = e.split(",")[0]
        #append_to_additional_file(e)
        da = e.split(",")[-1].split(";;")[0]
        da_encoding = create_one_hot_encoding(da, labels_dict)
        #if len(e.split(",")[-1].split(";;")) >1 :
        #    prev_da = e.split(",")[-1].split(";;")[1]
        #else:
        prev_da = None
        prev_da_encoding = create_one_hot_encoding(prev_da, labels_dict)
        texts.append(text)
        das.append(da_encoding)
        prev_das.append(prev_da_encoding)
    return [texts, np.asarray(prev_das), np.asarray(das)]

def create_labels_dict(training_data):
    labels_set = set()
    for e in training_data:
        text = e.split(",")[0]
        #append_to_additional_file(e)
        da = e.split(",")[-1].split(";;")[0]
        if da == "" or da == "\n":
            continue
        labels_set.add(da.lower())

    sorted_list = sorted(labels_set)
    labels_dict = {}
    for i, e in enumerate(sorted_list):
        labels_dict[e] = i
    return labels_dict


def load_data_and_labels_dialog_act(data_file, da_encoding_file="da_encoding.json", n_prev_da=1):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    examples = list(open(data_file, "r").readlines())
    if os.path.exists(da_encoding_file):
        with open(da_encoding_file, 'r') as in_file:
            labels_dict = json.load(in_file)
    else:
        labels_dict = create_labels_dict(examples)
        with open(da_encoding_file, 'w') as out_file:
            json.dump(labels_dict, out_file)

    return sample2text_prev_da(examples, labels_dict, n_prev_da)

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def batch_iter_da(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]