# CoSENT

Code for ***[CoSENT: Consistent Sentence Embedding via Similarity Ranking]***

## Overview
- sbert.py: main file for CoSENT.
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
