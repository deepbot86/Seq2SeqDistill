import argparse
from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
from transformers import BartConfig, T5Config
from datasets import load_dataset
from distill import DistillationTrainingArguments, DistillationTrainer
from transformers import DataCollatorForSeq2Seq

def load_teacher_model(model_type: str, local_path: str, model_name: str) -> BartForConditionalGeneration or T5ForConditionalGeneration:
    if model_type == 'bart':
        #check if local path is provided
        if local_path:
            model = BartForConditionalGeneration.from_pretrained(local_path)
        else:
            model = BartForConditionalGeneration.from_pretrained(model_name)
    elif model_type == 't5':
        if local_path:
            model = T5ForConditionalGeneration.from_pretrained(local_path)
        else:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        raise ValueError('model_type must be bart or t5')
    return model

def load_student_model(model_type: str, local_path: str, num_encoder_layers: str, num_decoder_layers:str, hidden_dim:int, vocab_size:str) -> BartForConditionalGeneration or T5ForConditionalGeneration:
    if model_type == 'bart':
        #check if local path is provided
        if local_path:
            student_model = BartForConditionalGeneration.from_pretrained(local_path)
        else:
            # load bart config
            # Define the configuration
            bart_config = BartConfig(
                d_model=hidden_dim,  # dimensions of the model
                encoder_layers=num_encoder_layers,  # number of encoder layers
                decoder_layers=num_decoder_layers,  # number of decoder layers
                vocab_size=vocab_size,  # vocabulary size
            )
            # Create the bart model from the configuration
            student_model = BartForConditionalGeneration(bart_config)

    elif model_type == 't5':
        if local_path:
            student_model = T5ForConditionalGeneration.from_pretrained(local_path)
        else:
            # load t5 config
            # Define the configuration
            t5_config = T5Config(
                d_model=hidden_dim,  # dimensions of the model
                encoder_layers=num_encoder_layers,  # number of encoder layers
                decoder_layers=num_decoder_layers,  # number of decoder layers
                vocab_size=vocab_size,  # vocabulary size
            )
            # Create the t5 model from the configuration
            student_model = T5ForConditionalGeneration(t5_config)
    else:
        raise ValueError('model_type must be bart or t5')
    return student_model

def load_tokenizer(model_type: str, local_path: str, model_name: str) -> AutoTokenizer:
    if model_type == 'bart':
        #check if local path is provided
        if local_path:
            tokenizer = AutoTokenizer.from_pretrained(local_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == 't5':
        if local_path:
            tokenizer = AutoTokenizer.from_pretrained(local_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError('model_type must be bart or t5')
    return tokenizer

def load_distill_dataset(dataset_name: str, local_path: str, dataset_data_type: str):
        if local_path:
            if dataset_data_type:
                dataset = load_dataset(dataset_data_type, data_files={'train': local_path})
            else:
                raise ValueError('dataset_data_type must be provided while using local dataset path')
        elif dataset_name:
            dataset = load_dataset(dataset_name)
        else:
            raise ValueError('dataset_name or dataset_local_path must be provided') 
        
        # split dataset into train and validation
        train_dataset = dataset['train'].train_test_split(test_size=0.1, shuffle=True)
        train_dataset = train_dataset['train']
        val_dataset = train_dataset['test']
        return train_dataset, val_dataset


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='parsing command line distillation arguments')
    # Add the arguments
    parser.add_argument('--model-type', required=True, type=str, help='currently only bart and t5 for conditional generation are supported')
    parser.add_argument('--teacher', type=str, help='huggingface teacher model name, e.g. facebook/bart-base, t5-base')
    parser.add_argument('--num-encoder-layers', type=str, default=3 ,help='student model number of encoder layers')
    parser.add_argument('--num-decoder-layers', type=str, default=3, help='student model number of decoder layers')
    parser.add_argument('--hidden-dim', type=str, default=512, help='huggingface student model hidden dimensions (d_model is 768 for bart-base and 512 for t5-base)')
    parser.add_argument('--vocab-size', type=str, default=50265, help='student model vocab size, default is 50265')
    parser.add_argument('--teacher-local-path', type=str, help='local path of huggingface teacher model name')
    parser.add_argument('--custom-tokenizer-local-path', type=float, help='custom tokenizer local path')
    parser.add_argument('--dataset', type=str, help='huggingface dataset name e.g. cnn_dailymail, samsum etc.')
    parser.add_argument('--dataset-input-column', type=str, required=True, help='dataset input column name')
    parser.add_argument('--dataset-target-column', type=str, required=True, help='dataset target column name')
    parser.add_argument('--dataset-local-path', type=str, help='local dataset path')
    parser.add_argument('--dataset-data-type', type=str, help='csv, json, etc. required if using local dataset path')
    parser.add_argument('--output-dir', type=str, help='output path of distilled strudent model')
    parser.add_argument('--max_length', type=int, default=512, help='maximum length of input sequence')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs for training')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='learning rate for training')
    parser.add_argument('--fp16', type=bool, default=True, help='use fp16 for training')
    parser.add_argument('--seed', type=int, default=42, help='random seed for training')
    parser.add_argument('--log-interval', type=int, default=10, help='log interval for training')
    parser.add_argument('--gradient-accumulation', type=bool, default=True, help='gradient accumulation for training')
    parser.add_argument('optimizer', type=str, default='adamw_torch', help='optimizer for training')
    # Parse the arguments
    args = parser.parse_args()

    # load tokenizer
    tokenizer = load_tokenizer(args.model_type, args.custom_tokenizer_local_path, args.teacher)

    vocab_size = tokenizer.vocab_size

    # load student model
    student_model = load_student_model(args.model_type, args.student_local_path, args.num_encoder_layers, args.num_decoder_layers, args.hidden_dim, vocab_size)

    # load teacher model
    teacher_model = load_teacher_model(args.model_type, args.teacher_local_path, args.teacher)

    # load dataset
    dataset = load_distill_dataset(args.dataset, args.dataset_local_path, args.dataset_data_type)

    # tokenizer dataset function
    # using closure for tokenizer since we would be using this function in generator and not in map function
    def preprocess_function(sample,
                        max_length: int,
                        input_col_name: str,
                        target_col_name: str,
                        padding: str="max_length"):
        """given input text dataset and tokenizer, tokenizer the text data for model training
            params:
               sample: text dataset element tp be tokenized
               tokenizer: AutoTokenizer - huggingface tokenizer object
               max_length: int - max sequence length
               padding - padding strategy, pad to max length 
               input_col_name: str - column with input data to model
               output_col_name: str - column with correct spelling label
           return - tokenized Dataset
        """

        # tokenize inputs
        model_inputs = tokenizer(sample[input_col_name], max_length=max_length, padding=padding, truncation=True)

        # tokenize labels/targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample[target_col_name], max_length=max_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # preprocess dataset
    # instead of using map function to tokenize dataset function we would use generators
    # https://huggingface.co/docs/datasets/processing.html#generators
    # this is important for processing large datasets, else we would get out of disk space errors 
    # since text to tokenization is space intensive (a char is 1 byte and a token is 4 bytes)
    # train_dataset = dataset['train'].map(
    #     preprocess_function,
    #     batched=True,
    #     remove_columns=[args.dataset_input_column, args.dataset_target_column],
    # )

    dataset.set_transform(preprocess_function)

    # set training arguments
    distill_training_args = DistillationTrainingArguments(
        output_dir=args.output_dir, # output directory
        num_train_epochs=args.epochs, # total number of training epochs
        per_device_train_batch_size=args.batch_size, # batch size per device during training
        per_device_eval_batch_size=args.batch_size, # batch size for evaluation
        warmup_steps=500, # number of warmup steps for learning rate scheduler
        weight_decay=0.01, # strength of weight decay
        logging_dir='./logs', # directory for storing logs
        logging_steps=10,
        learning_rate=args.learning_rate,
        eval_steps = args.logging_steps * 3,
        evaluation_strategy="steps",
        save_strategy = "no",
        optim = args.optimizer,
        fp16=args.fp16,
        # distilation parameters
        alpha=0.5,
        temperature=4.0,
        remove_unused_columns = False
    )

    # Data collator for Seq2Seq tasks: shifts the decoder input to the right by one position
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=student_model, return_tensors = "pt")

    # instantiate trainer
    trainer = DistillationTrainer(
        model=student_model, # the instantiated 🤗 Transformers model to be trained
        args=distill_training_args, # training arguments, defined above
        train_dataset=dataset["train"], # training dataset
        eval_dataset=dataset["validation"], # evaluation dataset
        data_collator=seq2seq_data_collator,
        teacher_model=teacher_model
    )

    # start training
    print('starting training')
    trainer.train()

    # save model
    print(f"Saving model on disk - {args.output_dir}")
    trainer.save_model(args.output_dir)


