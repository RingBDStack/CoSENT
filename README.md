# CoSENT

Code for ***CoSENT: Consistent Sentence Embedding via Similarity Ranking***

## Overview
- main.py: main file for CoSENT.
- datasets: dictionary contains all datasets used in our paper.
- loss.py: CoSENT loss.
- arguments.py: configurations for training and evaluation.
- utils.py: tools for reading datasets.
- SentEval.py and senteval: codes for SentEval tasks, senteval is copied from ***[SentEval: evaluation toolkit for sentence embeddings](https://github.com/facebookresearch/SentEval#senteval-evaluation-toolkit-for-sentence-embeddings)***


## Requirements
The implementation of CoSENT is tested under Python 3.9.13, with the following packages installed:
* `pytorch==1.12.0`
* `tqdm==4.64.0`
* `transformers==4.23.1`
* `sentence-transformers==2.2.2`
* `datasets==2.6.1`

## How to use CoSENT
We provide some example in parameters dictionary. 

Train $BERT{_\mathrm{base}}-CoSENT$ and $RoBERTa{_\mathrm{base}}-CoSENT$ models on STS-benchmark dataset:

```
python3 main.py parameters/bert_base_cosent_stsb.json
python3 main.py parameters/roberta_base_cosent_stsb.json
```

Pretrain $BERT{_\mathrm{base}}-CoSENT-NLI$ and $RoBERTa{_\mathrm{base}}-CoSENT-NLI$ models:

```
python3 main.py parameters/bert_base_cosent_nli.json
python3 main.py parameters/roberta_base_cosent_nli.json
```

You can also customize training via CLI:
```
python3  main.py --output_dir=output/BERT/base/paws_cosent_seq64_bt64 --dataset_name=paws --model_name_or_path=bert-base-uncased --pooling_type=mean --do_train --do_eval  --do_predict --loss_type=CoSENT --train_batch_size=64 --max_seq_length=64
```

Only evaluation fine-tuned models, specify your model path using parameter output_dir:
```
python3  main.py --output_dir=output/BERT/base/nli_cosent_first_last --dataset_name=stsb --model_name_or_path=bert-base-uncased --do_predict --loss_type=CoSENT
```
