# Task Specific Seq2Seq Model Distillation

This project is designed for Task Specific distillation of large Seq2Seq models like BART or T5 into smaller, more efficient models. The distilled models retain much of the performance of the original models while being faster and more memory efficient. The project supports distillation of both custom trained local checkpoints and pretrained checkpoints from Hugging Face.

The main.py script uses the DistillationTrainer class from the HuggingFace Transformers library to train a sequence-to-sequence model using knowledge distillation. It supports both BART and T5 models.

### Loss Function
The loss function used for distillation is a combination of Cross Entropy loss and KL Divergence loss. Cross Entropy loss measures the performance of a classification model whose output is a probability value between 0 and 1. KL Divergence loss measures how one probability distribution diverges from a second, expected probability distribution.
Weighted Loss Function 
    loss = alpha * student_loss + (1 - alpha) * kl_divergence_loss

## Installation

    pip install seq2seqdistill

    from seq2seqdistill.seq2seq_distill_trainer import Seq2SeqDistillTrainer
    training_args = {}
    training_args["model_type"] = "bart"
    training_args["teacher"] = "facebook/bart-base"
    training_args["dataset"] = "dataset"
    training_args["dataset"] = "samsum"
    training_args["dataset_input_column"] = "dialogue"
    training_args["dataset_target_column"] = "summary"
    training_args["output_dir"] = "distilled_bart_model_test"
    distiller = Seq2SeqDistillTrainer(training_args)
    distiller.train()

## Arguments

Here is a brief explanation of the arguments that can be passed from command line to main.py(if using Seq2SeqDistillTrainer then convert the "-" in argument name to '_" 
e.g. "model-type" to "model_type"):
    
    --model-type: The type of the model. Currently, only 'bart' and 't5' for conditional generation are supported. This argument is required.
    --teacher: The name of the teacher model, e.g., 'facebook/bart-base', 't5-base'.
    --num-encoder-layers: The number of encoder layers in the student model. Default is 3.
    --num-decoder-layers: The number of decoder layers in the student model. Default is 3.
    --hidden-dim: The hidden dimensions of the student model. Default is 512.
    --vocab-size: The vocab size of the student model. Default is 50265.
    --teacher-local-path: The local path of the teacher model.
    --custom-tokenizer-local-path: The local path of the custom tokenizer.
    --dataset: The name of the dataset, e.g., 'cnn_dailymail', 'samsum'.
    --dataset-input-column: The name of the input column in the dataset. This argument is required.
    --dataset-target-column: The name of the target column in the dataset. This argument is required.
    --dataset-local-path: The local path of the dataset.
    --dataset-data-type: The data type of the dataset, e.g., 'csv', 'json'. Required if using a local dataset path.
    --output-dir: The output path of the distilled student model.
    --max_length: The maximum length of the input sequence. Default is 512.
    --batch-size: The batch size for training. Default is 32.
    --epochs: The number of epochs for training. Default is 3.
    --learning-rate: The learning rate for training. Default is 5e-5.
    --fp16: Whether to use fp16 for training. Default is True.
    --seed: The random seed for training. Default is 42.
    --log-interval: The log interval for training. Default is 10.
    --gradient-accumulation: Whether to use gradient accumulation for training. Default is True.
    --optimizer: The optimizer for training. Default is 'adamw_torch'.
    

Please note that if you are using a custom trained model, you should provide the local path of the model. Similarly, if you are using a local dataset(or custom tokenizer), you should provide the local path of the dataset(tokenizer). 

## How to Run After Cloning Github Repo
    
    git clone https://github.com/deepbot86/Seq2SeqDistill.git

The main script for this project is Seq2SeqDistill/main.py. You can run this script from the command line with various arguments to specify the details of the distillation process. This script ha sbeen tested on samsum dataset for finetuning BART Base model on an AWS ml.p3.24xlarge instance (8 V100 GPUs) using torchrun. The distilled student model had 3 encoder and 3 decoder layers.  

Here are some examples:

### Distilling a BART model from Hugging Face
    python src/seq2seqdistill/main.py --model-type bart --teacher facebook/bart-base --num-encoder-layers 3 --num-decoder-layers 3 --hidden-dim 512 --vocab-size 50265 --dataset cnn_dailymail --dataset-input-column article --dataset-target-column highlights --dataset-local-path None --output-dir ./distilled_model

### Distilling a T5 model from Hugging Face
    python src/seq2seqdistill/main.py --model-type t5 --teacher t5-base --num-encoder-layers 3 --num-decoder-layers 3 --hidden-dim 512 --vocab-size 32128 --dataset cnn_dailymail --dataset-input-column article --dataset-target-column highlights --dataset-local-path None --output-dir ./distilled_model

### Distilling a custom trained BART model

    python src/seq2seqdistill/main.py --model-type bart --teacher-local-path /path/to/teacher/model --num-encoder-layers 3 --num-decoder-layers 3 --hidden-dim 512 --vocab-size 50265 --dataset cnn_dailymail --dataset-input-column article --dataset-target-column highlights --dataset-local-path /path/to/dataset --output-dir ./distilled_model

### Distilling a custom trained T5 model

    python src/seq2seqdistill/main.py --model-type t5 --teacher-local-path /path/to/teacher/model --num-encoder-layers 3 --num-decoder-layers 3 --hidden-dim 512 --vocab-size 32128 --dataset cnn_dailymail --dataset-input-column article --dataset-target-column highlights --dataset-local-path /path/to/dataset --output-dir ./distilled_model

### Distilling a custom trained BART model with custom trained tokenizer
    python src/seq2seqdistill/main.py --model-type bart --teacher facebook/bart-base --teacher-local-path /path/to/teacher/model --custom-tokenizer-local-path /path/to/custom/tokenizer --dataset samsum --dataset-input-column dialogue --dataset-target-column summary --output-dir /path/to/output/dir


# For distributed training using torchrun 
e.g. running the code on ml.p3.16xlarge instance that has 8 V100 GPUs, NUM_GPUS_YOU_HAVE should be set to 8
    
    torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE src/seq2seqdistill/main.py --model-type bart --teacher facebook/bart-base  --dataset samsum --dataset-input-column dialogue --dataset-target-column summary --output-dir /path/to/output/dir

## Dependencies

This project requires the following Python libraries:

- torch
- transformers
- datasets

To install dependencies 
    pip install -r requirements.txt


