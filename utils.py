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


def get_paws(loss_type):
    dataset = load_dataset('datasets/paws', 'labeled_final')
    train_samples = []
    dev_samples = []
    test_samples = []
    if loss_type == 'Softmax':
        for row in dataset['train']:
            inp_example = InputExample(
                texts=[row['sentence1'], row['sentence2']],
                label=int(row['label']))
            train_samples.append(inp_example)
        for row in dataset['validation']:
            inp_example = InputExample(
                texts=[row['sentence1'], row['sentence2']],
                label=int(row['label']))
            dev_samples.append(inp_example)
        for row in dataset['test']:
            inp_example = InputExample(
                texts=[row['sentence1'], row['sentence2']],
                label=int(row['label']))
            test_samples.append(inp_example)
    elif loss_type == 'CoSENT' or loss_type == 'CosineSimilarity' or loss_type == 'Contrastive':
        if loss_type == 'CosineSimilarity':
            logging.warning('Make sure the label is in 0-1')
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
    else:
        raise ValueError(f"unsupport loss type {loss_type}")
    return train_samples, dev_samples, test_samples


def get_stsb(loss_type):
    train_samples = []
    dev_samples = []
    test_samples = []
    dataset_path = 'datasets/stsbenchmark.tsv.gz'
    if not os.path.exists(dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz',
                      dataset_path)
    with gzip.open(dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if loss_type == 'CoSENT':
                score = float(row['score'])
            elif loss_type == 'CosineSimilarity':
                score = float(
                    row['score']) / 5.0  # Normalize score to range 0 ... 1
            elif loss_type == 'Contrastive':
                score = float(float(row['score']) > 2.5)
            elif loss_type == 'Softmax':
                # raise ValueError(
                #     'STSB is a score dataset, do not support softmax')
                score = int(float(row['score']))
            else:
                raise ValueError(f"unsupport loss type {loss_type}")
            # score = float(row['score'])
            inp_example = InputExample(
                texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)
    return train_samples, dev_samples, test_samples


def get_nli(loss_type):
    '''
    SNLI + MultiNLI (AllNLI) dataset
    '''
    train_samples = []
    dev_samples = []
    test_samples = []
    dataset_path = 'datasets/AllNLI.tsv.gz'
    if not os.path.exists(dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', dataset_path)
    label2int = {"contradiction": 0, "neutral": 1, "entailment": 2}
    with gzip.open(dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if loss_type == 'Softmax':
                label_id = label2int[row['label']]
            elif loss_type == 'CoSENT':
                label_id = float(label2int[row['label']])
            elif loss_type == 'CosineSimilarity':
                label_id = label2int[row['label']] / 2.0
            else:
                raise ValueError(f"unsupport loss type {loss_type}")
            inp_example = InputExample(
                texts=[row['sentence1'], row['sentence2']], label=label_id)
            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)
    return train_samples, dev_samples, test_samples


def get_sick(loss_type):
    dataset = load_dataset('datasets/sick')
    train_samples = []
    dev_samples = []
    test_samples = []

    if loss_type == 'CoSENT' or loss_type == 'CosineSimilarity' or loss_type == 'Contrastive':
        if loss_type == 'CosineSimilarity':
            logging.warning('Make sure the label is in 0-1')
            for row in dataset['train']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label=float(row['relatedness_score']) / 5)
                train_samples.append(inp_example)
            for row in dataset['validation']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label=float(row['relatedness_score']) / 5)
                dev_samples.append(inp_example)
            for row in dataset['test']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label=float(row['relatedness_score']) / 5)
                test_samples.append(inp_example)
        elif loss_type == 'CoSENT':
            for row in dataset['train']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label=float(row['relatedness_score']))
                train_samples.append(inp_example)
            for row in dataset['validation']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label=float(row['relatedness_score']))
                dev_samples.append(inp_example)
            for row in dataset['test']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label=float(row['relatedness_score']))
                test_samples.append(inp_example)
        elif loss_type == 'Contrastive':
            for row in dataset['train']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label=float(float(row['relatedness_score'])>3.0))
                train_samples.append(inp_example)
            for row in dataset['validation']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label=float(float(row['relatedness_score'])>3.0))
                dev_samples.append(inp_example)
            for row in dataset['test']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label=float(float(row['relatedness_score'])>3.0))
                test_samples.append(inp_example)
    else:
        raise ValueError(f"unsupport loss type {loss_type}")
    return train_samples, dev_samples, test_samples

def get_sick_nli(loss_type):
    dataset = load_dataset('datasets/sick')
    train_samples = []
    dev_samples = []
    test_samples = []

    if loss_type == 'CoSENT' or loss_type == 'CosineSimilarity' or loss_type == 'Softmax':
        if loss_type == 'CosineSimilarity':
            logging.warning('Make sure the label is in 0-1')
            for row in dataset['train']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label= (2 - float(row['label']))/2.0)
                train_samples.append(inp_example)
            for row in dataset['validation']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label= (2 - float(row['label']))/2.0)
                dev_samples.append(inp_example)
            for row in dataset['test']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label= (2 - float(row['label']))/2.0)
                test_samples.append(inp_example)
        elif loss_type == 'CoSENT' :
            for row in dataset['train']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label= 2.0 - float(row['label']))
                train_samples.append(inp_example)
            for row in dataset['validation']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label= 2.0 - float(row['label']))
                dev_samples.append(inp_example)
            for row in dataset['test']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label= 2.0 - float(row['label']))
                test_samples.append(inp_example)
        elif loss_type == 'Softmax' :
            for row in dataset['train']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label= 2 - int(row['label']))
                train_samples.append(inp_example)
            for row in dataset['validation']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label= 2 - int(row['label']))
                dev_samples.append(inp_example)
            for row in dataset['test']:
                inp_example = InputExample(
                    texts=[row['sentence_A'], row['sentence_B']],
                    label= 2 - int(row['label']))
                test_samples.append(inp_example)
    else:
        raise ValueError(f"unsupport loss type {loss_type}")
    return train_samples, dev_samples, test_samples

def get_sts(name, loss_type):
    dataset = load_dataset(f'datasets/{name}_sts')
    test_samples = []

    if loss_type in {'CoSENT', 'CosineSimilarity'}:
        if loss_type == 'CosineSimilarity':
            logging.warning('Make sure the label is in 0-1')
            for row in dataset['test']:
                inp_example = InputExample(
                    texts=[row['sentence1'], row['sentence2']],
                    label=float(row['score']) / 5)
                test_samples.append(inp_example)
        else:
            for row in dataset['test']:
                inp_example = InputExample(
                    texts=[row['sentence1'], row['sentence2']],
                    label=float(row['score']))
                test_samples.append(inp_example)
    else:
        raise ValueError(f"unsupport loss type {loss_type}")
    return test_samples

def get_atec(loss_type):
    train_samples = []
    dev_samples = []
    test_samples = []
    for splits in {'train', 'valid', 'test'}:
        with open(f'datasets/senteval_cn/ATEC/ATEC.{splits}.data', 'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                label = float(line[2]) if loss_type in {'CoSENT', 'CosineSimilarity', 'Contrastive'} else int(line[2])
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
        with open(f'datasets/senteval_cn/BQ/BQ.{splits}.data', 'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                label = float(line[2]) if loss_type in {'CoSENT', 'CosineSimilarity', 'Contrastive'} else int(line[2])
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
        with open(f'datasets/senteval_cn/LCQMC/LCQMC.{splits}.data', 'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                label = float(line[2]) if loss_type in {'CoSENT', 'CosineSimilarity'} else int(line[2])
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
        with open(f'datasets/senteval_cn/STS-B/STS-B.{splits}.data', 'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                if loss_type == 'Softmax':
                    raise ValueError('loss type not in {CoSENT, CosineSimilarity}')
                label = float(line[2])
                if loss_type == 'CosineSimilarity':
                    label = label / 5.0
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
    dataset = load_dataset('datasets/senteval_cn/pawsx', 'zh')
    train_samples = []
    dev_samples = []
    test_samples = []
    if loss_type == 'Softmax':
        for row in dataset['train']:
            inp_example = InputExample(
                texts=[row['sentence1'], row['sentence2']],
                label=int(row['label']))
            train_samples.append(inp_example)
        for row in dataset['validation']:
            inp_example = InputExample(
                texts=[row['sentence1'], row['sentence2']],
                label=int(row['label']))
            dev_samples.append(inp_example)
        for row in dataset['test']:
            inp_example = InputExample(
                texts=[row['sentence1'], row['sentence2']],
                label=int(row['label']))
            test_samples.append(inp_example)
    elif loss_type == 'CoSENT' or loss_type == 'CosineSimilarity' or loss_type == 'Contrastive':
        if loss_type == 'CosineSimilarity':
            logging.warning('Make sure the label is in 0-1')
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
    else:
        raise ValueError(f"unsupport loss type {loss_type}")
    return train_samples, dev_samples, test_samples


def get_mrpc(loss_type):
    dataset = load_dataset("datasets/mrpc")
    train_samples = []
    dev_samples = []
    test_samples = []
    if loss_type == 'Softmax':
        for row in dataset['train']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=int(row['label']))
            train_samples.append(inp_example)
        for row in dataset['validation']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=int(row['label']))
            dev_samples.append(inp_example)
        for row in dataset['test']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=int(row['label']))
            test_samples.append(inp_example)
    elif loss_type == 'CoSENT' or loss_type == 'CosineSimilarity' or loss_type == 'Contrastive':
        if loss_type == 'CosineSimilarity':
            logging.warning('Make sure the label is in 0-1')
        for row in dataset['train']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=float(row['label']))
            train_samples.append(inp_example)
        for row in dataset['validation']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=float(row['label']))
            dev_samples.append(inp_example)
        for row in dataset['test']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=float(row['label']))
            test_samples.append(inp_example)
    else:
        raise ValueError(f"unsupport loss type {loss_type}")
    return train_samples, dev_samples, test_samples

def get_rte(loss_type):
    dataset = load_dataset("SetFit/rte")
    train_samples = []
    dev_samples = []
    if loss_type == 'Softmax':
        for row in dataset['train']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=1 - int(row['label']))
            train_samples.append(inp_example)
        for row in dataset['validation']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=1 - int(row['label']))
            dev_samples.append(inp_example)
    elif loss_type == 'CoSENT' or loss_type == 'CosineSimilarity':
        if loss_type == 'CosineSimilarity':
            logging.warning('Make sure the label is in 0-1')
        for row in dataset['train']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=1 - float(row['label']))
            train_samples.append(inp_example)
        for row in dataset['validation']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=1 - float(row['label']))
            dev_samples.append(inp_example)
    else:
        raise ValueError(f"unsupport loss type {loss_type}")
    return train_samples, dev_samples, None

def get_qqp(loss_type):
    dataset = load_dataset("SetFit/qqp")
    train_samples = []
    dev_samples = []
    if loss_type == 'Softmax':
        for row in dataset['train']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=int(row['label']))
            train_samples.append(inp_example)
        for row in dataset['validation']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=int(row['label']))
            dev_samples.append(inp_example)
    elif loss_type == 'CoSENT' or loss_type == 'CosineSimilarity':
        if loss_type == 'CosineSimilarity':
            logging.warning('Make sure the label is in 0-1')
        for row in dataset['train']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=float(row['label']))
            train_samples.append(inp_example)
        for row in dataset['validation']:
            inp_example = InputExample(
                texts=[row['text1'], row['text2']],
                label=float(row['label']))
            dev_samples.append(inp_example)
    else:
        raise ValueError(f"unsupport loss type {loss_type}")
    return train_samples, dev_samples, None