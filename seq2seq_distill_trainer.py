from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
from transformers import BartConfig, T5Config
from datasets import load_dataset
from distill import DistillationTrainingArguments, DistillationTrainer
from transformers import DataCollatorForSeq2Seq
from utils import load_teacher_model, load_student_model, load_tokenizer, load_distill_dataset


class Seq2SeqDistillTrainer:
    def __init__(self, args):
        self.model_type = args.model_type
        if self.model_type == "bart":
            self.optimizer = "adamw_torch"
        elif self.model_type == "t5":
            self.optimizer = "adafactor"
        else:
            ValueError("model_type must be bart or t5")
        self.output_dir = args.output_dir
        self.num_train_epochs = args.epochs
        self.per_device_train_batch_size = args.batch_size
        self.per_device_eval_batch_size = args.batch_size
        self.evaluation_strategy = "steps"
        self.save_strategy = "no",
        self.optim = args.optimizer,
        self.fp16=args.fp16,
        # distilation parameters
        self.alpha=0.5,
        self.temperature=4.0,
        self.remove_unused_columns = False

        # load tokenizer
        self.tokenizer = load_tokenizer(args.model_type, args.custom_tokenizer_local_path, args.teacher)
        self.vocab_size = self.tokenizer.vocab_size
        # load student model
        self.student_model = load_student_model(args.model_type, args.num_encoder_layers, args.num_decoder_layers, args.hidden_dim, self.vocab_size)
        # load teacher model
        self.teacher_model = load_teacher_model(args.model_type, args.teacher_local_path, args.teacher)
        # load dataset
        self.dataset = load_distill_dataset(args.dataset, args.dataset_local_path, args.dataset_data_type)

        
        # adding generator for dataset tokenization, it will be performed on the fly at the time of training
        self.dataset.set_transform(self.preprocess_function)


    # tokenizer dataset function
    # using closure for tokenizer since we would be using this function in generator and not in map function
    def preprocess_function(self, 
                            sample,
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
        model_inputs = self.tokenizer(sample[input_col_name], max_length=max_length, padding=padding, truncation=True)

        # tokenize labels/targets with the `text_target` keyword argument
        labels = self.tokenizer(text_target=sample[target_col_name], max_length=max_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        

    def train(self):
        print('starting training')
        # set training arguments
        distill_training_args = DistillationTrainingArguments(
            output_dir=self.output_dir, # output directory
            num_train_epochs=self.epochs, # total number of training epochs
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
        
        # Data collator for Seq2Seq tasks: shifts the decoder input to the right by one position
        seq2seq_data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.student_model, return_tensors = "pt")

        # instantiate trainer
        trainer = DistillationTrainer(
            model=self.student_model, # the instantiated 🤗 Transformers model to be trained
            args=distill_training_args, # training arguments, defined above
            train_dataset=self.dataset["train"], # training dataset
            eval_dataset=self.dataset["validation"], # evaluation dataset
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