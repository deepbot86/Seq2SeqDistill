import argparse
from src.distillseq2seq.seq2seq_distill_trainer import Seq2SeqDistillTrainer
from src.distillseq2seq.utils import load_teacher_model, load_student_model, load_tokenizer, load_distill_dataset

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
    parser.add_argument('--max_length', type=int, default=128, help='maximum length of input sequence')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs for training')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='learning rate for training')
    parser.add_argument('--fp16', type=bool, default=True, help='use fp16 for training')
    parser.add_argument('--seed', type=int, default=42, help='random seed for training')
    parser.add_argument('--log-interval', type=int, default=10, help='log interval for training')
    parser.add_argument('--gradient-accumulation', type=bool, default=True, help='gradient accumulation for training')
    parser.add_argument('--optimizer', type=str, default='adamw_torch', help='optimizer for training')
    parser.add_argument("--local-rank", type=int, default=0)
    # Parse the arguments
    args = parser.parse_args()
    # covert arguments to dict
    args_dict = vars(args)
    seq2seq_trainer = Seq2SeqDistillTrainer(args_dict)
    seq2seq_trainer.train()

