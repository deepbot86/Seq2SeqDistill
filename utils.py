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
        labels = tokenizer(text_target=sample[output_col_name], max_length=args.max_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs