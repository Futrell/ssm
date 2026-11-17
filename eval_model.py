import os
import re
import sys
import csv
import glob
import math
import random
import argparse
import itertools
from typing import *
from collections import defaultdict

import tqdm
import torch
import numpy as np
import scipy

import process_data
import ssm
import convexity


CHECKPOINT_DIR = "checkpoints"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SHUFFLE_SEED = 0
DEV_PROPORTION = .1
INF = float('inf')

def deterministic_shuffle(xs, seed=SHUFFLE_SEED):
    ys = list(xs)
    random.seed(SHUFFLE_SEED)
    random.shuffle(ys)
    return ys

def train_dev_split(xs, proportion=DEV_PROPORTION):
    ys = deterministic_shuffle(xs)
    cutoff = math.floor(len(ys) * proportion)
    train = ys[:-cutoff]
    dev = ys[-cutoff:]
    return train, dev

def get_model(model_type: str,
              vocab_size: int,
              init_temperature: float,
              pi_offset: float = 0.0,
              state_dim: Optional[int] = None) -> ssm.PhonotacticsModel:
    if state_dim is None:
        state_dim = vocab_size + 1 # for eos

    if model_type == 'sl2':
        model = ssm.SL2.initialize(vocab_size, init_T=init_temperature)
    elif model_type == 'pfsa_sl2':
        # same as SL2, but backed by PFSA instead of SSM; outcomes should be identical
        model = ssm.pTSL.initialize(
            vocab_size,
            init_T=init_temperature,
            pi=torch.full((vocab_size-1,), INF, device=DEVICE), # force everything projected (eos always is)
            semiring=ssm.LogspaceSemiring,
        )
    elif model_type == 'sp2':
        model = ssm.SP2.initialize(vocab_size, init_T=init_temperature)
    elif model_type == 'sl2_times_pfsa':
        model1 = ssm.SL2.initialize(vocab_size, init_T=init_temperature)
        model2 = ssm.PFSAPhonotacticsModel.initialize(
            state_dim,
            vocab_size,
            init_T=init_temperature,
            semiring=ssm.LogspaceSemiring
        )
        model = model1 * model2
    elif model_type == 'ptsl2_times_pfsa':
        model1 = ssm.pTSL.initialize(vocab_size, init_T=init_temperature, pi_offset=pi_offset, semiring=ssm.LogspaceSemiring)
        model2 = ssm.PFSAPhonotacticsModel.initialize(
            state_dim,
            vocab_size,
            init_T=init_temperature,
            semiring=ssm.LogspaceSemiring
        )
        model = model1 * model2
    elif model_type == 'sl2_plus_pfsa':
        model1 = ssm.SL2.initialize(vocab_size, init_T=init_temperature)
        model2 = ssm.PFSAPhonotacticsModel.initialize(state_dim, vocab_size, init_T=init_temperature, semiring=ssm.LogspaceSemiring)
        model = model1 + model2
    elif model_type == 'ptsl2_plus_pfsa':
        model1 = ssm.pTSL.initialize(vocab_size, semiring=ssm.LogspaceSemiring, pi_offset=pi_offset)
        model2 = ssm.PFSAPhonotacticsModel.initialize(state_dim, vocab_size, init_T=init_temperature, semiring=ssm.LogspaceSemiring)
        model = model1 + model2
    elif model_type == 'sl2_times_sp2':
        model1 = ssm.SL2.initialize(vocab_size, init_T=init_temperature)
        model2 = ssm.SP2.initialize(vocab_size, init_T=init_temperature)
        model = model1 * model2
    elif model_type == 'quasi_sp2':
        model = ssm.QuasiSP2.initialize(vocab_size, init_T=init_temperature)
    elif model_type == 'soft_tsl2':
        model = ssm.SoftTSL2.initialize(
            vocab_size,
            init_T=init_temperature,
            init_T_projection=init_temperature,
            pi_offset=pi_offset,
        )
    elif model_type == 'sl2_times_soft_tsl2':
        model1 = ssm.SL2.initialize(vocab_size, init_T=init_temperature)
        model2 = ssm.SoftTSL2.initialize(
            vocab_size,
            init_T=init_temperature,
            init_T_projection=init_temperature,
            pi_offset=pi_offset,
        )
        model = model1 * model2
    elif model_type == 'sl2_times_ptsl2':
        model1 = ssm.SL2.initialize(vocab_size, init_T=init_temperature)
        model2 = ssm.pTSL.initialize(
            vocab_size,
            init_T=init_temperature,
            pi_offset=pi_offset,
            semiring=ssm.LogspaceSemiring,
        )
        model = model1 * model2
    elif model_type == "sl2_times_sp2_times_ptsl2":
        model1 = ssm.SL2.initialize(vocab_size, init_T=init_temperature)
        model2 = ssm.SP2.initialize(vocab_size, init_T=init_temperature)
        model3 = ssm.pTSL.initialize(
            vocab_size,
            init_T=init_temperature,
            pi_offset=pi_offset,
            semiring=ssm.LogspaceSemiring,
        )
        model = model1 * model2 * model3
    elif model_type == 'sl2_times_mtsl2':
        model1 = ssm.SL2.initialize(vocab_size, init_T=init_temperature)
        model2 = ssm.pTSL.initialize(
            vocab_size,
            init_T=init_temperature,
            pi_offset=pi_offset,
            semiring=ssm.LogspaceSemiring
        )
        model3 = ssm.pTSL.initialize(
            vocab_size,
            init_T=init_temperature,
            pi_offset=pi_offset,
            semiring=ssm.LogspaceSemiring
        )
        model = ssm.ProductPhonotacticsModel(model1, model2, model3)
    elif model_type == 'ptsl2':
        model = ssm.pTSL.initialize(vocab_size, init_T=init_temperature, pi_offset=pi_offset, semiring=ssm.LogspaceSemiring)
    elif model_type == 'double_ptsl2':
        model1 = ssm.pTSL.initialize(vocab_size, init_T=init_temperature, pi_offset=0.0, semiring=ssm.LogspaceSemiring)
        model2 = ssm.pTSL.initialize(vocab_size, init_T=init_temperature, pi_offset=pi_offset, semiring=ssm.LogspaceSemiring)
        model = model1 * model2
    elif model_type == "double_soft_tsl2":
        model1 = ssm.SoftTSL2.initialize(vocab_size, init_T=init_temperature, init_T_projection=init_temperature, pi_offset=0.0)
        model2 = ssm.SoftTSL2.initialize(vocab_size, init_T=init_temperature, init_T_projection=init_temperature, pi_offset=pi_offset)
        model = model1 * model2        
    elif model_type == 'ssm':
        model = ssm.SSMPhonotacticsModel.initialize(
            state_dim,
            vocab_size,
            init_T_A=init_temperature,
            init_T_B=init_temperature,
            init_T_C=init_temperature,
        )
    elif model_type == 'diag_ssm':
        model = ssm.SquashedDiagonalSSMPhonotacticsModel.initialize(
            state_dim,
            vocab_size,
            B=torch.eye(state_dim, device=DEVICE)[:, 1:],
            init_T_A=init_temperature,
            init_T_C=init_temperature,
        )
    elif model_type == 'double_diag_ssm':
        model1 = ssm.SquashedDiagonalSSMPhonotacticsModel.initialize(
            state_dim,
            vocab_size,
            B=torch.eye(state_dim, device=DEVICE)[:, 1:],
            init_T_A=init_temperature,
            init_T_C=init_temperature,
        )
        model2 = ssm.SquashedDiagonalSSMPhonotacticsModel.initialize(
            state_dim,
            vocab_size,
            B=torch.eye(state_dim, device=DEVICE)[:, 1:],
            init_T_A=init_temperature,
            init_T_C=init_temperature,
        )
        model = model1 * model2
    elif model_type == 'sl2_times_diag_ssm':
        model1 = ssm.SL2.initialize(vocab_size, init_T=init_temperature)
        model2 = ssm.SquashedDiagonalSSMPhonotacticsModel.initialize(
            state_dim,
            vocab_size,
            B=torch.eye(state_dim, device=DEVICE)[:, 1:],
            init_T_A=init_temperature,
            init_T_C=init_temperature,
        )
        model = model1 * model2
    elif model_type == 'pfsa':
        model = ssm.PFSAPhonotacticsModel.initialize(state_dim, vocab_size, init_T=init_temperature, semiring=ssm.LogspaceSemiring)
    elif model_type == 'opfsa':
        model = ssm.OverparameterizedPFSAPhonotacticsModel.initialize(state_dim, 2*state_dim, vocab_size, init_T=init_temperature, semiring=ssm.LogspaceSemiring)
    elif model_type == 'wfsa':
        model = ssm.WFSAPhonotacticsModel.initialize(state_dim, vocab_size, init_T=init_temperature, learn_final=True)
    else:
        raise ValueError("Unrecognized model type: %s" % model_type)

    return model.to(DEVICE)

def get_vocab_size(data):
    # Each symbol is represented by an integer, starting from 0. To get vocab
    # size, find the largest integer used and add 1. Potentially includes EOS.
    good_vocab_size = max(map(max, data)) + 1
    return good_vocab_size

def numerical_eval(test_data, judgments):
    judgments = np.array(list(map(float, judgments)))
    def compute_correlations(model):
        scores = model.log_likelihood(test_data, debug=False).detach().cpu().numpy()
        return {
            'spearman': scipy.stats.spearmanr(scores, judgments).statistic,
            'pearson': scipy.stats.pearsonr(scores, judgments).statistic,
        }
    return compute_correlations

def is_numeric_column(values) -> bool:
    """Numeric if every value parses as a number and there are more than two
    distinct values; a two-valued column (e.g. 0/1) is treated as categories."""
    try:
        floats = [float(v) for v in values]
    except ValueError:
        return False
    return len(set(floats)) > 2

def label_column_evals(test_data, columns, names, numeric_first=False):
    """Evaluation for every label column of a test file: correlations for a
    numeric column, per-category scores (plus a diff and t-statistic when
    there are two categories) otherwise. The first column's metrics keep
    their plain names (spearman, TRUE_scores, ...) for compatibility with
    existing outputs; later columns are prefixed with their header name."""
    evals = []
    for i, (column, name) in enumerate(zip(columns, names)):
        numeric = (numeric_first and i == 0) or is_numeric_column(column)
        fn = numerical_eval(test_data, column) if numeric else categorical_eval(test_data, column)
        prefix = '' if i == 0 else name + '_'
        print("Test column %s: %s evaluation" % (name, 'numerical' if numeric else 'categorical'), file=sys.stderr)
        evals.append((prefix, fn))

    def compute(model):
        result = {}
        for prefix, fn in evals:
            result |= {prefix + k: v for k, v in fn(model).items()}
        return result
    return compute

def dev_eval(dev_data):
    def do_dev_eval(model):
        result = {}
        result['dev_loss'] = -model.log_likelihood(dev_data).mean().detach().cpu().numpy()
        return result
    return do_dev_eval

def categorical_eval(test_data, judgments, paired=False):
    categorized_data = defaultdict(list)
    for form, value in zip(test_data, judgments):
        categorized_data[value].append(form)
    values = list(categorized_data.keys())
    if not paired:
        # Paired data keeps the file's column order; otherwise sort so the
        # sign of the diff doesn't depend on which row happens to come first.
        values = sorted(values)

    def compute_scores(model):
        result = {}
        scores = [
            model.log_likelihood(categorized_data[value]).detach().cpu().numpy()
            for value in values
        ]
        for value, score in zip(values, scores):
            result['%s_scores' % value] = score.mean().item()
        if len(scores) == 2:
            label = '%s_%s' % (str(values[0]), str(values[1]))
            if paired:
                # Same items in both categories: mean pairwise diff, paired t-test.
                result[label + '_diff'] = (scores[0] - scores[1]).mean().item()
                result[label + '_diff_t'] = scipy.stats.ttest_rel(*scores).statistic
            else:
                # Different items: diff of means, Welch's unequal-variance t-test.
                result[label + '_diff'] = (scores[0].mean() - scores[1].mean()).item()
                result[label + '_diff_t'] = scipy.stats.ttest_ind(*scores, equal_var=False).statistic
        return result

    return compute_scores

def checkpoint_step(path: str) -> int:
    """Training step encoded in a checkpoint filename of the form <prefix>_<step>.pt."""
    match = re.search(r'_(\d+)\.pt$', path)
    if match is None:
        raise ValueError("Cannot parse step from checkpoint filename: %s" % path)
    return int(match.group(1))

REPORTED_HYPERPARAMS = ['model_type', 'batch_size', 'lr']

def filename_hyperparams(path: str) -> Dict:
    """Fallback for checkpoints saved before models carried a .hyperparams
    attribute: parse the run scripts' naming scheme,
    <model_type>_bs<N>[_ep<E>]_lr<lr>_<step>.pt. Returns whatever was found."""
    name = os.path.basename(path)
    result = {}
    model_type = re.match(r'(.+?)_bs\d+_', name)
    if model_type:
        result['model_type'] = model_type.group(1)
    bs = re.search(r'_bs(\d+)(?=_)', name)
    if bs:
        result['batch_size'] = int(bs.group(1))
    lr = re.search(r'_lr([0-9.eE+-]+)_\d+\.pt$', name)
    if lr:
        try:
            result['lr'] = float(lr.group(1))
        except ValueError:
            pass
    return result

def checkpoint_hyperparams(model, path: str, overrides: Optional[Dict]=None) -> Dict:
    """Hyperparameters to report for a checkpoint. Precedence: explicit
    overrides > the model's saved .hyperparams > the filename."""
    saved = getattr(model, 'hyperparams', None) or {}
    overrides = overrides or {}
    for k, v in overrides.items():
        if k in saved and saved[k] != v:
            print("Warning: %s: --%s=%s overrides saved value %s" % (path, k, v, saved[k]), file=sys.stderr)
    merged = filename_hyperparams(path) | saved | overrides
    if 'model_type' not in merged:
        raise ValueError("%s: cannot determine model_type from checkpoint or filename; pass --model_type" % path)
    # Same column order as training output.
    return {k: merged[k] for k in REPORTED_HYPERPARAMS if k in merged}

def evaluate_checkpoints(paths: Iterable[str],
                         eval_fn: Callable[[torch.nn.Module], Dict],
                         phone2ix: Dict,
                         train_data=None,
                         convexity_check: bool=False,
                         hyperparams_to_report: Optional[Dict]=None,
                         param_deltas: bool=False,
                         param_delta_norm: Optional[str]=None,
                         outfile=sys.stdout):
    """Load each saved checkpoint and run eval_fn on it, writing CSV rows
    in the same format as PhonotacticsModel.train (minus mean_loss, which
    depended on training batches; train_loss over the full train set is
    reported instead when train_data is given). With param_deltas, also
    reports how far each parameter moved since the previous checkpoint."""
    paths = sorted(paths, key=checkpoint_step)
    writer = csv.writer(outfile)
    first_time = True
    prev_params = prev_step = None
    for path in tqdm.tqdm(paths):
        # Checkpoints are whole pickled models, so weights_only must be False.
        model = torch.load(path, map_location=DEVICE, weights_only=False)
        model_phone2ix = getattr(model, 'phone2ix', None)
        if model_phone2ix is not None and model_phone2ix != phone2ix:
            raise ValueError("%s: checkpoint vocabulary does not match the training file's" % path)
        diagnostic = {'step': checkpoint_step(path), 'checkpoint': path}
        hyperparams = checkpoint_hyperparams(model, path, hyperparams_to_report)
        with torch.no_grad():
            diagnostic |= eval_fn(model)
            if train_data is not None:
                diagnostic['train_loss'] = -model.log_likelihood(train_data).mean().item()
        if param_deltas:
            step = checkpoint_step(path)
            cur_params = ssm.param_snapshot(model)
            diagnostic['steps_since_prev'] = None if prev_step is None else step - prev_step
            diagnostic |= ssm.compute_param_deltas(prev_params, cur_params, norm=param_delta_norm)
            prev_params, prev_step = cur_params, step
        diagnostic |= hyperparams
        if convexity_check:
            loss_closure = lambda: -model.log_likelihood(train_data).mean()
            low, high = convexity.hessian_spectrum(loss_closure, list(model.parameters()))
            diagnostic['nonconvexity'] = abs(min(0, low)) / abs(high)
            diagnostic['high_eigval'] = high
            diagnostic['low_eigval'] = low
            bigger = max(abs(high), abs(low))
            smaller = min(abs(high), abs(low))
            diagnostic['condition_number'] = bigger / smaller
        if first_time:
            writer.writerow(list(diagnostic.keys()))
            first_time = False
        writer.writerow(list(diagnostic.values()))
        outfile.flush()

if __name__ == "__main__":
    # With --eval_checkpoints, the model type is normally read from the
    # checkpoint, so it becomes an optional --model_type override instead of
    # the first positional argument.
    eval_mode = any(
        a == '--eval_checkpoints' or a.startswith('--eval_checkpoints=')
        for a in sys.argv[1:]
    )
    parser = argparse.ArgumentParser(description="Train a phonotactics model, or evaluate saved checkpoints.")
    if eval_mode:
        parser.add_argument('--model_type', type=str, default=None,
                            help="Override the model type saved in the checkpoints")
    else:
        parser.add_argument('model_type', type=str, help="Model type, a string")
    parser.add_argument('train_file', type=str,
                        help="Training data, consisting only of good forms")
    parser.add_argument('test_file', type=str, default=None,
                        help="""Test data, two columns where first is a form
                                and second is a judgment (TRUE or FALSE, or numerical)""")
    parser.add_argument('--dev_file', type=str, default=None, help="Held out training forms")
    parser.add_argument('--dev_split', action='store_true', help="Whether to hold out %s of train data randomly as a dev set, if no dev file is specified" % str(DEV_PROPORTION))
    parser.add_argument('--no_header', action='store_true', help="Whether train/dev files have no header")
    parser.add_argument('--test_no_header', action='store_true',
                        help="""Whether the test file has no header. Without this flag the
                                first line of the test file is treated as a header.""")
    parser.add_argument('--test_data_paired', action='store_true',
                        help="Whether the test data is in two-column paired grammatical/ungrammatical format.")
    parser.add_argument('--numerical_eval', action='store_true',
                        help="""Force numerical evaluation of the first test label column.
                                Normally unnecessary: each label column is evaluated as
                                numerical or categorical based on its values.""")
    parser.add_argument('--char_separator', type=str, default=" ",
                        help="Delimiter for characters")
    parser.add_argument('--col_separator', type=str, default="\t",
                        help="Delimiter for forms vs. judgments in data files")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Batch size (default 1; with --eval_checkpoints, overrides the saved value)")
    parser.add_argument('--num_epochs', type=int, default=1,
                        help="Number of passes through the training data")
    parser.add_argument('--lr', type=float, default=None,
                        help="Adam learning rate (default .001; with --eval_checkpoints, overrides the saved value)")
    parser.add_argument('--convexity_check', action='store_true',
                        help="Whether to compute a convexity metric at each step (slow)")
    parser.add_argument('--init_temperature', type=float, default=100,
                        help="Initialization temperature")
    parser.add_argument('--pi_offset', type=float, default=0.0,
                        help="Offset for projection logit initialization")
    parser.add_argument('--report_every', type=int, default=1,
                        help="How often to report training results")
    parser.add_argument('--reporting_window_size', type=int, default=1,
                        help="Window size for averaging for reporting loss")
    parser.add_argument('--save_checkpoints', action='store_true',
                   help="Whether to save checkpoints.")
    parser.add_argument('--checkpoint_folder', type=str, default='.')
    parser.add_argument('--checkpoint_filename', type=str, default="model",
                        help="Filename prefix for checkpoints.")
    parser.add_argument('--eval_checkpoints', type=str, default=None,
                        help="""Glob pattern of saved checkpoints (quote it!). If given,
                                skip training and evaluate each checkpoint instead. Usage:
                                eval_model.py TRAIN_FILE TEST_FILE --eval_checkpoints GLOB""")
    parser.add_argument('--param_deltas', action='store_true',
                        help="""Report, for each parameter, the L2 distance from its value at
                                the previous report step (or previous checkpoint, with
                                --eval_checkpoints), as columns delta_<name>""")
    parser.add_argument('--param_delta_norm', choices=ssm.PARAM_DELTA_NORMS, default=None,
                        help="""Also report normalized deltas: rms = L2/sqrt(numel),
                                relative = L2/||previous value||""")
    parser.add_argument('--no_filter_training_grammatical', action='store_true',
                        help="Do NOT filter training data to only include grammatical examples.")

    args = parser.parse_args()
    if args.test_no_header and args.test_data_paired:
        parser.error("--test_data_paired needs the test file's header: it names the two columns")
    if args.param_delta_norm is not None and not args.param_deltas:
        parser.error("--param_delta_norm requires --param_deltas")

    phone2ix, train_data, _ = process_data.process_data(
        args.train_file,
        col_separator=args.col_separator,
        char_separator=args.char_separator,
        header=not args.no_header,
        filter_grammatical=not args.no_filter_training_grammatical  # Default to True unless disabled
    )
    vocab_size = get_vocab_size(train_data)

    if args.test_file is None:
        test_eval_fn = None
    else:
        _, test_data, test_extra = process_data.process_data(
            args.test_file,
            col_separator=args.col_separator,
            char_separator=args.char_separator,
            phone2ix=phone2ix, # ensure same vocabulary mapping
            paired=args.test_data_paired,
            header=not args.test_no_header,
        )
        if args.test_data_paired:
            # Each row is a pair of forms; labels come from the header.
            test_eval_fn = categorical_eval(test_data, test_extra[0], paired=True)
        else:
            # Every column after the form is a label column, evaluated by type.
            if args.test_no_header:
                names = ['label%d' % (i + 1) for i in range(len(test_extra))]
            else:
                with open(args.test_file, newline='') as f:
                    header = next(csv.reader(f, delimiter=args.col_separator))
                names = header[1:1 + len(test_extra)]
            test_eval_fn = label_column_evals(test_data, test_extra, names, numeric_first=args.numerical_eval)

    if args.dev_file is None and not args.dev_split:
        dev_eval_fn = None
    elif args.dev_split:
        train_data, dev_data = train_dev_split(train_data)
        dev_eval_fn = dev_eval(dev_data)
        print(
            "Split into train set with %d examples and dev set with %d examples" % (len(train_data), len(dev_data)),
            file=sys.stderr,
        )
    else:
        _, dev_data, _ = process_data.process_data(
            args.dev_file,
            col_separator=args.col_separator,
            char_separator=args.char_separator,
            phone2ix=phone2ix,
            header=not args.no_header,
        )
        dev_eval_fn = dev_eval(dev_data)

    def eval_fn(model):
        d = {}
        if dev_eval_fn is not None:
            d |= dev_eval_fn(model)
        if test_eval_fn is not None:
            d |= test_eval_fn(model)
        return d

    if args.eval_checkpoints is not None:
        paths = glob.glob(args.eval_checkpoints)
        if not paths:
            sys.exit("No checkpoints match %s" % args.eval_checkpoints)
        print("Evaluating %d checkpoints" % len(paths), file=sys.stderr)
        evaluate_checkpoints(
            paths,
            eval_fn,
            phone2ix,
            train_data=train_data,
            convexity_check=args.convexity_check,
            param_deltas=args.param_deltas,
            param_delta_norm=args.param_delta_norm,
            # Only explicitly given values; the rest come from each checkpoint.
            hyperparams_to_report={
                k: v for k, v in [
                    ('model_type', args.model_type),
                    ('batch_size', args.batch_size),
                    ('lr', args.lr),
                ] if v is not None
            },
        )
        sys.exit(0)

    if args.batch_size is None:
        args.batch_size = 1
    if args.lr is None:
        args.lr = .001

    batches = tqdm.tqdm(list(ssm.minibatches(train_data, args.batch_size, args.num_epochs)))
    model = get_model(args.model_type, vocab_size, init_temperature=args.init_temperature, pi_offset=args.pi_offset)
    model.phone2ix = phone2ix # just attach the phoneme-to-index mapping to the model
    # Also attach the hyperparameters, so checkpoints are self-describing.
    model.hyperparams = {
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'init_temperature': args.init_temperature,
        'pi_offset': args.pi_offset,
    }
    if not args.save_checkpoints:
        checkpoint_prefix = None
    else:
        os.makedirs(os.path.join(args.checkpoint_folder, 'checkpoints'), exist_ok=True)
        checkpoint_prefix = "%s/checkpoints/%s" % (args.checkpoint_folder, args.checkpoint_filename)

    model.train(
        batches,
        report_every=args.report_every,
        reporting_window_size=args.reporting_window_size,
        lr=args.lr,
        eval_fn=eval_fn,
        convexity_check=args.convexity_check,
        checkpoint_prefix=checkpoint_prefix,
        hyperparams_to_report={k: model.hyperparams[k] for k in REPORTED_HYPERPARAMS},
        param_deltas=args.param_deltas,
        param_delta_norm=args.param_delta_norm,
    )
