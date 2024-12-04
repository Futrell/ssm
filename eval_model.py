import csv
import argparse

import ssm
import process_data

def get_model(model_type: str, vocab_size: int):
    if model_type == 'sl2':
        return ssm.SL2.initialize(vocab_size)
    else:
        crash

parser = argparse.ArgumentParser(description="Train a phonotactics model.")
parser.add_argument('model_type', type=str, help="Model type, a string")
parser.add_argument('train_file', type=str, help="Training data, only TRUE cases will be used")
parser.add_argument('--char_separator', type=str, default="", help="Delimiter for characters")
parser.add_argument('--col_separator', type=str, default="\t", help="Delimiter for forms vs. judgments in data files")
parser.add_argument('--test_file', type=str, default=None, help="Test data, two columns where first is a form and second is a judgment")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
parser.add_argument('--num_epochs', type=int, default=1, help="Number of iterations through training data")
parser.add_argument('--report_every', type=int, default=1, help="How often to report training results")
parser.add_argument('--reporting_window_size', type=int, default=100, help="Window size for averaging for reporting diagnostics")

args = parser.parse_args()
*_, data = process_data.process_data(
    args.train_file,
    col_separator=args.col_separator,
    char_separator=args.char_separator
)
vocab_size = max(map(max, data)) + 1
# TODO: Read in test data, use it to formulate good vs. bad eval to be passed in as as diagnostic fn
batches = ssm.minibatches(data, args.batch_size, args.num_epochs)
model = get_model(args.model_type, vocab_size)
model.train(
    batches,
    report_every=args.report_every,
    reporting_window_size=args.reporting_window_size
)



