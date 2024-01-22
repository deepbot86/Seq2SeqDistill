from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
from transformers import BartConfig, T5Config
from datasets import load_dataset
from src.distillseq2seq.distill import DistillationTrainingArguments, DistillationTrainer
from transformers import DataCollatorForSeq2Seq
from src.distillseq2seq.utils import load_teacher_model, load_student_model, load_tokenizer, load_distill_dataset


class Seq2SeqDistillTrainer:
    def __init__(self, args):
        self.model_type = args.get("model_type", None)
        self.teacher = args.get("teacher", None)
        self.teacher_local_path = args.get("teacher_local_path", None)
        self.dataset = args.get("dataset", None)
        self.dataset_local_path = args.get("dataset_local_path", None)
        self.dataset_data_type = args.get("dataset-data-type", None)
        self.custom_tokenizer_local_path = args.get("custom_tokenizer_local_path", None)
        # if teacher model's local path is provided and custom_tokenizer_local_path is not provided then use teacher_local_path as custom_tokenizer_local_path
        if self.teacher_local_path:
            if not self.custom_tokenizer_local_path:
                self.custom_tokenizer_local_path = self.teacher_local_path
        if self.model_type == "bart":
            self.optimizer = "adamw_torch"
        elif self.model_type == "t5":
            self.optimizer = "adafactor"
        else:
            ValueError("model_type must be bart or t5")
        self.output_dir = args.get("output_dir", None)
        self.input_col_name = args.get("dataset_input_column", None)
        self.target_col_name = args.get("dataset_target_column", None)
        self.max_length = args.get("max_length", 128)
        self.num_train_epochs = args.get("epochs", 3)
        self.hidden_dim = args.get("hidden_dim", 512)
        self.num_encoder_layers = args.get("num_encoder_layers", 3)
        self.num_decoder_layers = args.get("num_decoder_layers", 3)
        self.batch_size = args.get("batch_size", 8)
        self.per_device_train_batch_size = args.get("batch_size", 8)
        self.per_device_eval_batch_size = args.get("batch_size", 8)
        self.evaluation_strategy = "steps"
        self.save_strategy = "no"
        self.fp16 = args.get("fp16", True)
        # distilation parameters
        self.alpha=0.5
        self.temperature=4.0
        self.remove_unused_columns = False
        
        assert self.model_type is not None, "model_type should not be None, it should be either 'bart' or 't5'"
        assert self.teacher is not None or self.teacher_local_path is not None, "both teacher and teacher_local_path can't be None"
        assert self.input_col_name is not None, "input_col_name argumentcannot be None - please provide input column name in dataset"
        assert self.target_col_name is not None, "provide target_col_name argument cannot be None - please provide target column name in dataset"
        assert self.dataset is not None or self.dataset_local_path is not None, "both dataset and dataset_local_path cannot be None, either provide dataset name from Huggingface datasets or provide path of local dataset saved on disk"
        assert self.output_dir is not None, "output_dir cannot be None, provide the name of directory where distilled model will be saved "
        
        if not self.dataset:
            assert self.dataset_data_type is not None, "dataset_data_type cannot be None when dataset argument is None, it should be e.g. csv, txt etc"

        

        # load tokenizer
        self.tokenizer = load_tokenizer(self.model_type, self.custom_tokenizer_local_path, self.teacher)
        self.vocab_size = self.tokenizer.vocab_size
        # load student model
        self.student_model = load_student_model(self.model_type, self.num_encoder_layers, self.num_decoder_layers, self.hidden_dim, self.vocab_size)
        # load teacher model
        self.teacher_model = load_teacher_model(self.model_type, self.teacher_local_path, self.teacher)
        # load dataset
        self.train_dataset, self.validation_dataset = load_distill_dataset(self.dataset, self.dataset_local_path, self.dataset_data_type)

        
        # adding generator for dataset tokenization, it will be performed on the fly at the time of training
        self.train_dataset.set_transform(self.preprocess_function)
        self.validation_dataset.set_transform(self.preprocess_function)


    # tokenizer dataset function
    # using closure for tokenizer since we would be using this function in generator and not in map function
    def preprocess_function(self, 
                            sample):
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
        model_inputs = self.tokenizer(sample[self.input_col_name], max_length=self.max_length, padding="max_length", truncation=True)

        # tokenize labels/targets with the `text_target` keyword argument
        labels = self.tokenizer(text_target=sample[self.target_col_name], max_length=self.max_length, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        

    def train(self):
        print('starting training')
        # set training arguments
        distill_training_args = DistillationTrainingArguments(
            output_dir=self.output_dir, # output directory
            num_train_epochs=self.num_train_epochs, # total number of training epochs
            per_device_train_batch_size=self.batch_size, # batch size per device during training
            per_device_eval_batch_size=self.batch_size, # batch size for evaluation
            warmup_steps=500, # number of warmup steps for learning rate scheduler
            weight_decay=0.01, # strength of weight decay
            logging_dir='./logs', # directory for storing logs
            logging_steps=100,
            learning_rate=5e-5,
            eval_steps = 200,
            evaluation_strategy="steps",
            save_strategy = "no",
            optim = self.optimizer,
            fp16=self.fp16,
            # distilation parameters
            alpha=0.5,
            temperature=4.0,
            remove_unused_columns = False
        )
        
        print("Using Following Training Arguments For Student")
        print(f"num_train_epochs = {self.num_train_epochs}")
        print(f"per_device_train_batch_size = {self.batch_size}")
        print("learning_rate = 5e-5")
        print(f"fp16 = {self.fp16}")
        print(f"Student Model Hidden Dim = {self.hidden_dim}")
        print(f"Student Model num_encoder_layers = {self.num_encoder_layers}")
        print(f"Student Model num_decoder_layers = {self.num_decoder_layers}")
        print(f"Student Model vocab_size = {self.tokenizer.vocab_size}")
        
        # Data collator for Seq2Seq tasks: shifts the decoder input to the right by one position
        seq2seq_data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.student_model, return_tensors = "pt")

        # instantiate trainer
        self.trainer = DistillationTrainer(
            model=self.student_model, # the instantiated 🤗 Transformers model to be trained
            args=distill_training_args, # training arguments, defined above
            train_dataset=self.train_dataset, # training dataset
            eval_dataset=self.validation_dataset, # evaluation dataset
            data_collator=seq2seq_data_collator,
            teacher_model=self.teacher_model
        )
        # start training
        self.trainer.train()
        # save model
        self.save_model() 

    def save_model(self):
        print(f"Saving model on disk - {self.output_dir}")
        self.trainer.save_model(self.output_dir)