from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task: str = field(
        default="summarization",
        metadata={
            "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
            "pegasus)"
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    line_by_line: bool = field(
        default=True,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    source_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    target_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    source_raw_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    source_file_random: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    target_raw_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    opinions_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        # default=1024,
        default=100,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        # default=128,
        default=100,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )


@dataclass
class DataValidationArguments:
    """
    Arguments pertaining to what parameters we are going to input to our model for validation.
    """
    min_summ_length: Optional[int] = field(
        default=100,
        metadata={
            "help": "The minimum length of the sequence to be generated."
        },
    )
    max_summ_length: Optional[int] = field(
        default=300,
        metadata={
            "help": "The maximum length of the sequence to be generated."
        },
    )

    num_beams: Optional[int] = field(
        default=3,
        metadata={
            "help": "Number of beams for beam search. 1 means no beam search."
        },
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences."
        },
    )

    no_repeat_ngram_size: Optional[int] = field(
        default=2,
        metadata={
            "help": " If set to int > 0, all ngrams of that size can only occur once."
        },
    )