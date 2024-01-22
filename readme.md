# Task Specific Seq2Seq Model Distillation

This project is designed for Task Specific distillation of large Seq2Seq models like BART or T5 into smaller, more efficient models. The distilled models retain much of the performance of the original models while being faster and more memory efficient. The project supports distillation of both custom trained local checkpoints and pretrained checkpoints from Hugging Face.

The main.py script uses the DistillationTrainer class from the HuggingFace Transformers library to train a sequence-to-sequence model using knowledge distillation. It supports both BART and T5 models.

### Loss Function
The loss function used for distillation is a combination of Cross Entropy loss and KL Divergence loss. Cross Entropy loss measures the performance of a classification model whose output is a probability value between 0 and 1. KL Divergence loss measures how one probability distribution diverges from a second, expected probability distribution.
Weighted Loss Function 
    loss = alpha * student_loss + (1 - alpha) * kl_divergence_loss

## How to Run

The main script for this project is Seq2SeqDistill/main.py. You can run this script from the command line with various arguments to specify the details of the distillation process.

Here are some examples:

### Distilling a BART model from Hugging Face
    python main.py --model-type bart --teacher facebook/bart-base --num-encoder-layers 3 --num-decoder-layers 3 --hidden-dim 512 --vocab-size 50265 --dataset cnn_dailymail --dataset-input-column article --dataset-target-column highlights --dataset-local-path None --output-dir ./distilled_model

### Distilling a T5 model from Hugging Face
    python main.py --model-type t5 --teacher t5-base --num-encoder-layers 3 --num-decoder-layers 3 --hidden-dim 512 --vocab-size 32128 --dataset cnn_dailymail --dataset-input-column article --dataset-target-column highlights --dataset-local-path None --output-dir ./distilled_model

### Distilling a custom trained BART model

    python main.py --model-type bart --teacher-local-path /path/to/teacher/model --num-encoder-layers 3 --num-decoder-layers 3 --hidden-dim 512 --vocab-size 50265 --dataset cnn_dailymail --dataset-input-column article --dataset-target-column highlights --dataset-local-path /path/to/dataset --output-dir ./distilled_model

### Distilling a custom trained T5 model

    python main.py --model-type t5 --teacher-local-path /path/to/teacher/model --num-encoder-layers 3 --num-decoder-layers 3 --hidden-dim 512 --vocab-size 32128 --dataset cnn_dailymail --dataset-input-column article --dataset-target-column highlights --dataset-local-path /path/to/dataset --output-dir ./distilled_model

### Distilling a custom trained BART model with custom trained tokenizer
    python main.py --model-type bart --teacher facebook/bart-base --teacher-local-path /path/to/teacher/model --custom-tokenizer-local-path /path/to/custom/tokenizer --dataset samsum --dataset-input-column source --dataset-target-column target --output-dir /path/to/output/dir

## Arguments

Here is a brief explanation of the arguments:

--model-type: The type of the model. Currently, only 'bart' and 't5' are supported.
--teacher: The Hugging Face model name of the teacher model.
--teacher-local-path: The local path of the teacher model.
--custom-tokenizer-local-path: The local path of the custom tokenizer.
--num-encoder-layers: The number of encoder layers in the student model.
--num-decoder-layers: The number of decoder layers in the student model.
--hidden-dim: The hidden dimensions of the student model.
--vocab-size: The vocabulary size of the student model.
--dataset: The Hugging Face dataset name.
--dataset-input-column: The input column name in the dataset.
--dataset-target-column: The target column name in the dataset.
--dataset-local-path: The local path of the dataset.
--output-dir: The output directory of the distilled model.  

Please note that if you are using a custom trained model, you should provide the local path of the model and set the corresponding Hugging Face model name to None. Similarly, if you are using a local dataset, you should provide the local path of the dataset and set the Hugging Face dataset name to None.

# For distributed training using torchrun 
e.g. running the code on ml.p3.16xlarge instance that has 8 V100 GPUs, NUM_GPUS_YOU_HAVE should be set to 8
    
    torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE main.py --model-type bart --teacher facebook/bart-base  --dataset samsum --dataset-input-column source --dataset-target-column target --output-dir /path/to/output/dir

## Dependencies

This project requires the following Python libraries:

- torch
- transformers
- datasets

To install dependencies 
    pip install -r requirements.txt





