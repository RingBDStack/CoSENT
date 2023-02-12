from typing import Optional
from dataclasses import dataclass, field
import logging
import math


@dataclass
class DataTraingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: str = field(
        metadata={
            "help":
            "The name of the dataset to use (via huggingface datasets library)"
        })
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use"})
    num_labels: Optional[int] = field(
        default=None,
        metadata={'help': 'number of the categories of dataset label'})
    # max_seq_length: int = field(
    #     default=768,
    #     metadata={
    #         "help":
    #         ("The maximum total input sequence length after tokenization. Sequences longer "
    #          "than this will be truncated, sequences shorter will be padded.")
    #     })
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached preprocessed datasets or not."
        })

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("dataset name can't be None")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            'help':
            'path to pretrained model or model identifier from huggingface.co/models'
        })
    pooling_type: str = field(
        metadata={'help': 'pooling layer type: cls, mean,  max, or first-last'})
    loss_type: str = field(
        default='CoSENT',
        metadata={'help': 'loss type: CosineSimilarity, CoSENT, Softmax'})
    config_name: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Pretrained config name or path if not the same as model_name'
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Pretrained tokenizer name or path if not the same as model_name'
        })
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'Truncate any inputs longer than max_seq_length'
        }
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("model name or path can't be None")
        if self.pooling_type is None:
            raise ValueError("pooling type can't be None")
        else:
            assert self.pooling_type in {'cls', 'mean', 'max', 'first-last'}


@dataclass
class TrainArguments:
    '''
    Arguments for train, validate, test
    do_eval: if not set, we will save the final model, else we will save the best model according to evaluator
    evaluation_step: if is 0 and do_eval, we will call evaluator after each epoch
    '''
    output_dir: str = field(
        metadata={
            "help":
            "The output directory where the model predictions and checkpoints will be written."
        })
    seed: int = field(default=42, metadata={'help': 'set seed for torch'})
    train_batch_size: int = field(default=16,
                                  metadata={'help': 'train batch size'})
    num_train_epochs: int = field(default=4,
                                  metadata={'help': 'train epoch numbers'})
    resume_output_dir: bool = field(
        default=False,
        metadata={
            "help":
            ("Resume the content of the output directory. "
             "Use this to continue training if output_dir points to a checkpoint directory."
             )
        },
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={
            'help':
            ('Linear warmup over warmup_ratio fraction of total steps.')
        })
    warmup_steps: int = field(
        default=0, metadata={'help': 'Linear warmup over warmup_steps.'})
    evaluation_steps: int = field(default=1000,
                                  metadata={'help': 'evaluate step in train'})
    do_train: bool = field(default=False,
                           metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(
        default=False,
        metadata={"help": "Whether to run predictions on the test set."})
    show_progress_bar: Optional[bool] = field(
        default=True, metadata={'help': 'If True, output a tqdm progress bar'})

    def __post_init__(self):
        if self.output_dir is None:
            raise ValueError("output_dir can't be None")
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logging.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio"
                " during training")

    def get_warmup_steps(self, num_training_steps: int):
        '''
        Get number of steps used for a linear warmup.
        num_training_steps include all epochs
        '''
        warmup_steps = (self.warmup_steps if self.warmup_steps > 0 else
                        math.ceil(num_training_steps * self.warmup_ratio))
        return warmup_steps
