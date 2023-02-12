'''
Author: Samrito
Date: 2022-11-12 17:28:19
LastEditors: Samrito
LastEditTime: 2022-12-07 10:00:37
'''
import datasets
from sentence_transformers.readers import InputExample
from sentence_transformers import util
import gzip, csv, os
import logging
from datasets.load import load_dataset

def get_atec(loss_type):
    train_samples = []
    dev_samples = []
    test_samples = []
    for splits in {'train', 'valid', 'test'}:
        with open(f'../datasets/senteval_cn/ATEC/ATEC.{splits}.data', 'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                label = float(line[2])
                inp_example = InputExample(
                    texts=[line[0], line[1]],
                    label=label)
                if splits == 'train':
                    train_samples.append(inp_example)
                elif splits == 'valid':
                    dev_samples.append(inp_example)
                elif splits == 'test':
                    test_samples.append(inp_example)
    return train_samples, dev_samples, test_samples

def get_bq(loss_type):
    train_samples = []
    dev_samples = []
    test_samples = []
    for splits in {'train', 'valid', 'test'}:
        with open(f'../datasets/senteval_cn/BQ/BQ.{splits}.data', 'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                label = float(line[2])
                inp_example = InputExample(
                    texts=[line[0], line[1]],
                    label=label)
                if splits == 'train':
                    train_samples.append(inp_example)
                elif splits == 'valid':
                    dev_samples.append(inp_example)
                elif splits == 'test':
                    test_samples.append(inp_example)
    return train_samples, dev_samples, test_samples

def get_lcqmc(loss_type):
    train_samples = []
    dev_samples = []
    test_samples = []
    for splits in {'train', 'valid', 'test'}:
        with open(f'../datasets/senteval_cn/LCQMC/LCQMC.{splits}.data', 'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                label = float(line[2])
                inp_example = InputExample(
                    texts=[line[0], line[1]],
                    label=label)
                if splits == 'train':
                    train_samples.append(inp_example)
                elif splits == 'valid':
                    dev_samples.append(inp_example)
                elif splits == 'test':
                    test_samples.append(inp_example)
    return train_samples, dev_samples, test_samples





def get_stsb_cn(loss_type):
    train_samples = []
    dev_samples = []
    test_samples = []
    for splits in {'train', 'valid', 'test'}:
        with open(f'../datasets/senteval_cn/STS-B/STS-B.{splits}.data', 'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                label = float(line[2])/5.0
                inp_example = InputExample(
                    texts=[line[0], line[1]],
                    label=label)
                if splits == 'train':
                    train_samples.append(inp_example)
                elif splits == 'valid':
                    dev_samples.append(inp_example)
                elif splits == 'test':
                    test_samples.append(inp_example)
    return train_samples, dev_samples, test_samples


def get_pawsx(loss_type):
    dataset = load_dataset('../datasets/senteval_cn/pawsx', 'zh')
    train_samples = []
    dev_samples = []
    test_samples = []
    for row in dataset['train']:
        inp_example = InputExample(
            texts=[row['sentence1'], row['sentence2']],
            label=float(row['label']))
        train_samples.append(inp_example)
    for row in dataset['validation']:
        inp_example = InputExample(
            texts=[row['sentence1'], row['sentence2']],
            label=float(row['label']))
        dev_samples.append(inp_example)
    for row in dataset['test']:
        inp_example = InputExample(
            texts=[row['sentence1'], row['sentence2']],
            label=float(row['label']))
        test_samples.append(inp_example)
    return train_samples, dev_samples, test_samples


