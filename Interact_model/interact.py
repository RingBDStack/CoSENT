'''
Author: Samrito
Date: 2022-12-07 10:00:05
LastEditors: Samrito
LastEditTime: 2022-12-07 10:00:06
'''
import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.readers import InputExample
from datasets.load import load_dataset
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
from arguments import DataTraingArguments, ModelArguments, TrainArguments
from loss import CoSENTLoss

from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
import utils

if __name__ == '__main__':
    parser = HfArgumentParser(
        (ModelArguments, DataTraingArguments, TrainArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        )
    if model_args.loss_type == 'Softmax':
        if data_args.num_labels is None:
            raise ValueError(
                "num labels need to be defined when loss is Softmax")

    torch.manual_seed(training_args.seed)
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print train arguments
    logging.info(f"*** dataset parameters ***\n {data_args}")
    logging.info(f"*** model parameters ***\n {model_args}")
    logging.info(f"*** Training/evaluation parameters ***\n {training_args}")

    if data_args.dataset_name in {'paws', 'stsb', 'nli', 'sick', 'sick_nli' ,'atec', 'bq','lcqmc', 'pawsx', 'stsb_cn','mrpc', 'rte', 'qqp'}:
        f = getattr(utils, f'get_{data_args.dataset_name}')
        train_samples, dev_samples, test_samples = f(model_args.loss_type)
    elif data_args.dataset_name in {
            'sts12', 'sts13', 'sts14', 'sts15', 'sts16'
    }:
        assert training_args.do_train == False and training_args.do_eval == False and training_args.do_predict == True
        test_samples = utils.get_sts(data_args.dataset_name,
                                     model_args.loss_type)
    else:
        raise ValueError('dataset name is not allowed')

    model_name = model_args.model_name_or_path

    if training_args.do_train:
        logging.info('*** Train ***')
        model = CrossEncoder(model_args.model_name_or_path, num_labels = data_args.num_labels, max_length = model_args.max_seq_length)
            # Convert the dataset to a DataLoader ready for training
        logging.info(f"Read {data_args.dataset_name} train dataset")

        train_dataloader = DataLoader(
            train_samples,
            shuffle=True,
            batch_size=training_args.train_batch_size)
        # train_loss = losses.CosineSimilarityLoss(model=model)
        if model_args.loss_type == 'CoSENT':
            train_loss = CoSENTLoss(model=model)
        else:
            train_loss = None

        logging.info(f"Read {data_args.dataset_name} dev dataset")
        evaluator = CECorrelationEvaluator.from_input_examples(
            dev_samples, name=f'{data_args.dataset_name}-dev')

        # Configure the training. We skip evaluation in this example
        warmup_steps = training_args.get_warmup_steps(
            len(train_dataloader) * training_args.num_train_epochs)
        logging.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        if training_args.do_eval:
            model.fit(train_dataloader = train_dataloader,
                      loss_fct = train_loss,
                      evaluator=evaluator,
                      epochs=training_args.num_train_epochs,
                      evaluation_steps=training_args.evaluation_steps,
                      warmup_steps=warmup_steps,
                      show_progress_bar=training_args.show_progress_bar,
                      output_path=training_args.output_dir)
        else:
            model.fit(train_dataloader = train_dataloader,
                      loss_fct = train_loss,
                      epochs=training_args.num_train_epochs,
                      warmup_steps=warmup_steps,
                      show_progress_bar=training_args.show_progress_bar,
                      output_path=training_args.output_dir)

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on test dataset
    #
    ##############################################################################
    if training_args.do_predict:
        logging.info('*** Evaluation ***')
        model = CrossEncoder(training_args.output_dir)
        # model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
        test_evaluator = CECorrelationEvaluator.from_input_examples(
            test_samples, name=f'{data_args.dataset_name}-test')
        test_evaluator(model, output_path=training_args.output_dir)
