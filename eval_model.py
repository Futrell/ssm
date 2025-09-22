import os
import csv
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


CHECKPOINT_DIR = "checkpoints"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

INF = float('inf')

def get_model(model_type: str,
              vocab_size: int,
              init_temperature: float,
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
            pi=torch.full((vocab_size,), INF, device=DEVICE), # force everything projected
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
        model1 = ssm.pTSL.initialize(vocab_size, init_T=init_temperature, semiring=ssm.LogspaceSemiring)
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
        model1 = ssm.pTSL.initialize(vocab_size, semiring=ssm.LogspaceSemiring)
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
        )
    elif model_type == 'sl2_times_soft_tsl2':
        model1 = ssm.SL2.initialize(vocab_size, init_T=init_temperature)
        model2 = ssm.SoftTSL2.initialize(
            vocab_size,
            init_T=init_temperature,
            init_T_projection=init_temperature,
        )        
        model = model1 * model2
    elif model_type == 'sl2_times_ptsl2':
        model1 = ssm.SL2.initialize(vocab_size, init_T=init_temperature)
        model2 = ssm.pTSL.initialize(
            vocab_size,
            init_T=init_temperature,
            semiring=ssm.LogspaceSemiring,
        )
        model = model1 * model2
    elif model_type == 'sl2_times_mtsl2':
        model1 = ssm.SL2.initialize(vocab_size, init_T=init_temperature)
        model2 = ssm.pTSL.initialize(
            vocab_size,
            init_T=init_temperature,
            semiring=ssm.LogspaceSemiring
        )
        model3 = ssm.pTSL.initialize(
            vocab_size,
            init_T=init_temperature,
            semiring=ssm.LogspaceSemiring
        )
        model = ssm.ProductPhonotacticsModel(model1, model2, model3)
    elif model_type == 'ptsl2':
        model = ssm.pTSL.initialize(vocab_size, init_T=init_temperature, semiring=ssm.LogspaceSemiring)
    elif model_type == 'double_ptsl2':
        model1 = ssm.pTSL.initialize(vocab_size, init_T=init_temperature, semiring=ssm.LogspaceSemiring)
        model2 = ssm.pTSL.initialize(vocab_size, init_T=init_temperature, semiring=ssm.LogspaceSemiring)
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
    elif model_type == 'diag_ssm_flex': 
        model = ssm.SquashedDiagonalSSMPhonotacticsModel.initialize(
            state_dim,
            vocab_size,
            init_T_A=init_temperature,
            init_T_B=init_temperature,
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
            #B=torch.eye(state_dim, device=DEVICE)[:, 1:],
            init_T_A=init_temperature,
            init_T_B=init_temperature,
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

    def compute_scores(model):
        result = {}
        scores = [
            model.log_likelihood(categorized_data[value]).detach().cpu().numpy()
            for value in values
        ]
        for value, score in zip(values, scores):
            result['%s_scores' % value] = score.mean().item()
        if len(scores) == 2 and paired:
            diffs = scores[0] - scores[1]
            result['%s_%s_diff' % (str(values[0]), str(values[1]))] = diffs.mean().item()
            result['%s_%s_diff_t' % (str(values[0]), str(values[1]))] = scipy.stats.ttest_rel(*scores).statistic
        return result
            
    return compute_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a phonotactics model.")
    parser.add_argument('model_type', type=str, help="Model type, a string")
    parser.add_argument('train_file', type=str,
                        help="Training data, consisting only of good forms")
    parser.add_argument('test_file', type=str, default=None,
                        help="""Test data, two columns where first is a form
                                and second is a judgment (TRUE or FALSE, or numerical)""")
    parser.add_argument('--dev_file', type=str, default=None, help="Held out training forms")
    parser.add_argument('--no_header', action='store_true', help="Whether train/dev files have no header")
    parser.add_argument('--test_data_paired', action='store_true',
                        help="Whether the test data is in two-column paired grammatical/ungrammatical format.")
    parser.add_argument('--numerical_eval', action='store_true',
                        help="Whether test judgments are numerical, rather than categorical.")
    parser.add_argument('--char_separator', type=str, default=" ",
                        help="Delimiter for characters")
    parser.add_argument('--col_separator', type=str, default="\t",
                        help="Delimiter for forms vs. judgments in data files")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=1,
                        help="Number of iterations through training data")
    parser.add_argument('--lr', type=float, default=.001,
                        help="Adam learning rate")
    parser.add_argument('--convexity_check', action='store_true',
                        help="Whether to compute a convexity metric at each step (slow)")
    parser.add_argument('--init_temperature', type=float, default=100,
                        help="Initialization temperature")
    parser.add_argument('--report_every', type=int, default=1,
                        help="How often to report training results")
    parser.add_argument('--reporting_window_size', type=int, default=1,
                        help="Window size for averaging for reporting loss")
    parser.add_argument('--save_checkpoints', action='store_true',
                        help="Whether to save checkpoints.")
    parser.add_argument('--checkpoint_folder', type=str, default='.')
    parser.add_argument('--checkpoint_filename', type=str, default="model",
                        help="Filename prefix for checkpoints.")

    args = parser.parse_args()

    phone2ix, train_data, train_extra = process_data.process_data(
        args.train_file,
        col_separator=args.col_separator,
        char_separator=args.char_separator,
        header=not args.no_header,
    ) # TODO: need to filter true?
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
        )
        judgments = test_extra[0]
        if args.numerical_eval:
            test_eval_fn = numerical_eval(test_data, judgments)
        else:
            test_eval_fn = categorical_eval(test_data, judgments, paired=args.test_data_paired)

    if args.dev_file is None:
        dev_eval_fn = None
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
        
    batches = tqdm.tqdm(list(ssm.minibatches(train_data, args.batch_size, args.num_epochs + 1)))
    model = get_model(args.model_type, vocab_size, init_temperature=args.init_temperature)
    model.phone2ix = phone2ix # just attach the phoneme-to-index mapping to the model
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
        hyperparams_to_report={
            'batch_size': args.batch_size,
            'lr': args.lr,
        }
    )
