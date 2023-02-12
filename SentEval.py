import os
import torch
import numpy as np
from tqdm import tqdm
import scipy.stats
import senteval
import logging
from prettytable import PrettyTable
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_LENGTH = 256
BATCH_SIZE = 512
TEST_PATH = 'datasets/senteval'
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

def sent_to_vec(sent, tokenizer, model, pooling, max_length):
 
    with torch.no_grad():
        inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True,  max_length=max_length)
        # inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(DEVICE)
        inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

        hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

        if pooling == 'first_last_avg':
            output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        elif pooling == 'mean':
            output_hidden_state = (hidden_states[-1]).mean(dim=1)
        elif pooling == 'last2avg':
            output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
        elif pooling == 'cls':
            output_hidden_state = (hidden_states[-1])[:, 0, :]
            
        else:
            raise Exception("unknown pooling {}".format(POOLING))

        vec = output_hidden_state.cpu().numpy()[0]
    return vec

MODEL_ZOOS = {
    'BERTbase-first_last_avg': {
        'encoder': 'output/BERT_1epoch/base/nli_cosent_first_last_seq64_bt64',
        'pooling': 'first_last_avg'
    },

    'BERTlarge-first_last_avg': {
        'encoder': 'output/BERT_1epoch/large/nli_cosent_first_last_seq64_bt64',
        'pooling': 'first_last_avg'
    },
}

def prepare(params, samples):
    return None


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = []
    for sent in batch:
        vec = sent_to_vec(sent, params['tokenizer'], \
                params['encoder'], params['pooling'], MAX_LENGTH)
        embeddings.append(vec)
    embeddings = np.vstack(embeddings)
    return embeddings
 

def run(model_name, test_path):

    model_config = MODEL_ZOOS[model_name]
    logging.info(f"{model_name} configs: {model_config}")

    tokenizer = AutoTokenizer.from_pretrained(model_config['encoder'])
    encoder = AutoModel.from_pretrained(model_config['encoder'])
    encoder = encoder.to(DEVICE)
    logging.info("Building {} tokenizer and model successfuly.".format(model_config['encoder']))

    # Set params for senteval
    params_senteval = {
            'task_path': test_path,
            'usepytorch': True,
            'tokenizer': tokenizer,
            'encoder': encoder,
            'pooling': 'last2avg',
            'batch_size': BATCH_SIZE,
            'kfold': 10       }

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = [
            'MR', 'CR', 'SUBJ', 'MPQA', 'SST2',
            'TREC', 
            'MRPC'
        ]
    results = se.eval(transfer_tasks)
    
    print(results)


def run_all_model():

    for model_name in MODEL_ZOOS:
        run(model_name, TEST_PATH)


if __name__ == "__main__":
    # run('BERTbase-first_last_avg', TEST_PATH)
    run_all_model()
