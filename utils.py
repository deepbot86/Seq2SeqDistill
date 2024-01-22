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

def load_student_model(model_type: str, num_encoder_layers: str, num_decoder_layers:str, hidden_dim:int, vocab_size:str) -> BartForConditionalGeneration or T5ForConditionalGeneration:
    if model_type == 'bart':
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
        split_dataset = dataset['train'].train_test_split(test_size=0.1, shuffle=True)
        print(split_dataset)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
        return train_dataset, val_dataset
