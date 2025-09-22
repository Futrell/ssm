import sys
import csv
import random
from typing import *
import operator
import functools
import itertools
from collections import namedtuple, deque

import torch
import pandas as pd
import numpy as np
import torch_semiring_einsum as tse

import naturalgradient

INF = float('inf')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_INIT_TEMPERATURE = 100
EPSILON = 10 ** -10

class Semiring:
    @classmethod
    def vv(self, x, y):
        multiplied = self.mul(x, y)
        return self.sum(multiplied, dim=-1)

    @classmethod
    def mv(self, A, x):
        multiplied = self.mul(A, x[None, :])
        return self.sum(multiplied, dim=-1)

    @classmethod
    def mm(self, A, B):
        multiplied = self.mul(A[:, :, None], B[None, :, :])
        return self.sum(multiplied, dim=-2)

class BooleanSemiring(Semiring):
    zero = False
    one = True
    add = operator.or_
    mul = operator.and_
    sum = torch.any
    prod = torch.all
    einsum = torch.einsum
    complement = operator.invert
    from_exp = lambda x:x
    to_exp = lambda x:x
    to_log = torch.log

    
class RealSemiring(Semiring):
    zero = 0.0
    one = 1.0
    
    add = operator.add
    mul = operator.mul
    div = operator.truediv
    pow = operator.pow
    
    sum = torch.sum
    prod = torch.prod
    vv = operator.matmul
    mv = operator.matmul
    mm = operator.matmul
    from_exp = lambda x:x
    from_log = torch.exp
    to_exp = lambda x:x
    to_log = torch.log
    logistic = torch.sigmoid
    einsum = torch.einsum

    @classmethod
    def complement(cls, x):
        return 1-x

class LogspaceSemiring(Semiring):
    zero = -INF
    one = 0.0
    
    mul = operator.add
    div = operator.sub
    pow = operator.mul

    prod = torch.sum    
    
    from_exp = torch.log
    from_log = lambda x:x
    to_exp = torch.exp
    to_log = lambda x:x
    logistic = torch.nn.functional.logsigmoid

    add = torch.logaddexp
    sum = torch.logsumexp

    @classmethod
    def add(cls, x, y):
        # logaddexp that preserves gradients when both x and y contain -inf
        Z = x.exp() + y.exp()                       
        tiny = torch.finfo(Z.dtype).tiny                # e.g., ~1e-45 for float32
        logZ = torch.log(torch.clamp_min(Z, tiny))      # avoid  log(0)
        result = torch.where(Z == 0, torch.full_like(Z, -torch.inf), logZ)  # put back -inf
        return result

    @classmethod
    def sum(cls, x: torch.Tensor, dim, keepdim=False):
        # logsumexp that preserves gradients when x contains -inf
        mask = torch.isfinite(x)
        neg_big = torch.finfo(x.dtype).min / 4
        x_finite = torch.where(mask, x, torch.full_like(x, neg_big))
        y = torch.logsumexp(x_finite, dim=dim, keepdim=True)
        all_invalid = ~mask.any(dim=dim, keepdim=True)
        y = torch.where(all_invalid, torch.full_like(y, -torch.inf), y)
        return y if keepdim else y.squeeze(dim)

    @classmethod
    def einsum(cls, formula, *args):
        return tse.log_einsum(tse.compile_equation(formula), *args)

    @classmethod
    def complement(cls, x):
        # numerically stable log(1-exp(x))
        return (-x.exp()).log1p()
    

SSMOutput = namedtuple("SSMOutput", "u proj x y".split())

def plot_ssm_output(x: SSMOutput, pi: Optional[torch.Tensor]=None, cmap='viridis'):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each array using imshow
    axs[0].imshow(x.u.detach().numpy().T, cmap=cmap)
    axs[0].set_title('Input')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Input dimension')

    axs[1].imshow(x.x.detach().numpy().T, cmap=cmap)
    axs[1].set_title('State')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('State dimension')

    if x.x.shape[-1] == x.u.shape[-1] + 1:
        x_coord, y_coord = np.where(x.u == 1)
        if pi is not None:
            indices = list((x.u.long() @ torch.arange(x.u.shape[-1])).numpy())
            colors = list(pi.sigmoid()[indices].detach().numpy())
            axs[1].scatter(x_coord, y_coord+1, c=colors, cmap='plasma', marker='o')
        else:
            axs[1].scatter(x_coord, y_coord+1, color='red', marker='o')

    axs[2].imshow(x.proj.detach().numpy().T, cmap=cmap)
    axs[2].set_title('Projection Probability')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('State dimension')

    # Display the plot
    plt.tight_layout()
    plt.show()

class WFSA(torch.nn.Module):
    def __init__(self, A, init=None, final=None, semiring=None):
        super().__init__()
        self.A = A # positive weights
        self.device = DEVICE

        X1, Y1, X2 = self.A.shape
        assert X1 == X2

        self.dtype = self.A.dtype
        if semiring is None:
            self.semiring = BooleanSemiring if self.dtype is torch.bool else RealSemiring
        else:
            self.semiring = semiring

        if init is None: # default to [1, 0, 0, 0, ...], enforcing state 0 = initial state.
            self.init = self.semiring.from_exp(torch.nn.functional.one_hot(torch.tensor(0), X1)).to(DEVICE)
        else:
            self.init = init

        if final is None: # default to [..., 1, 1, 1]
            self.final = torch.full_like(self.init, self.semiring.one)
        else:
            self.final = final

    def forward(self, input, debug=False):
        x = self.init
        # y = init A_1 A_2 A_3 ... A_T final
        for u_t in input:
            A_t = self.A[:, u_t, :]
            x = self.semiring.mv(A_t.T, x) 
        y = self.semiring.vv(x, self.final)
        if debug:
            breakpoint()
        return y

    def forward_incremental(self, input, debug=False):
        Q, S, R = self.A.shape
        A_reshaped = self.A.view(Q, -1).T
        def gen():
            q = self.init
            for t, u_t in enumerate(input):
                y = self.semiring.mv(A_reshaped, q).view(S, R)
                s = self.semiring.sum(y, -1) # vector of values for outputs
                yield s
                q = self.semiring.div(y[u_t], s[u_t])
                if debug:
                    breakpoint()
        return torch.stack(list(gen()))

    def transition_closure(self):
        # do calculation in exp space
        A = self.semiring.to_exp(self.semiring.sum(self.A, -2))
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        result = torch.linalg.solve(I - A, I)
        return self.semiring.from_exp(result)

    def state_occupancy(self):
        """ Expected state occupancy counts (always real semiring) """
        P = self.semiring.to_exp(self.semiring.sum(self.A, -2))
        init = self.semiring.to_exp(self.init)    
        I = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
        result = torch.linalg.solve(I - P.T, init)
        return result
    
    def pathsum(self):
        A_star = self.transition_closure()
        occupancy = self.semiring.mv(A_star.T, self.init)
        return self.semiring.vv(occupancy, self.final)

    def apply_fisher(self, v):
        """ Apply the Fisher information matrix to v """
        p = self.semiring.to_exp(self.A) # hack
        mask = p > 0.0
        Q, S, Q2 = p.shape        
        vA = torch.where(mask, v.view_as(self.A), 0.0)
        
        p_sum = p.sum((-1,-2), keepdim=True)
        p_proj = torch.where(mask, p / p_sum, 0.0)
        
        sum_v = vA.sum((-1,-2), keepdim=True)
        v_tan = vA - p_proj * sum_v
        
        w = torch.where(mask, v_tan / p_proj, 0.0)
        sum_w = w.sum((-1,-2), keepdim=True)
        w_tan = w - p_proj * sum_w

        # Scale by nonnegative expected occupancy counts
        counts = self.state_occupancy()
        result = counts[:, None, None] * w_tan
        return result


def softmax_fisher(A):
    # F_ij = p_i \delta_ij - p_i p_j
    #      = p_i (\delta_ij - p_j)
    I = torch.eye(A.shape[-1])
    F = A.unsqueeze(-1) * (I - A.unsqueeze(-2))
    return F

def test_wfsa():
    # pfsa for *bc
    A = torch.Tensor([
        # from state 0
        [[1/3, 0], # a
         [0, 1/3], # b
         [1/3, 0]], # c
        # from state 1
        [[1/2, 0], # a
         [1/2, 0], # b
         [0, 0]], # c
    ])
    fsa = WFSA(A, init=torch.eye(2, device=DEVICE)[0], final=torch.ones(2, device=DEVICE))
    assert fsa([1,2,0]) == 0
    assert fsa([1,0,2]) > 0

    # wfsa for *bc
    A = torch.Tensor([
        # from state 0
        [[1, 0], # a
         [0, 1], # b
         [1, 0]], # c
        # from state 1
        [[1, 0], # a
         [1, 0], # b
         [0, 0]], # c
    ]) / 2.74
    fsa = WFSA(A, init=torch.eye(2, device=DEVICE)[0], final=torch.ones(2, device=DEVICE))
    assert fsa([1,2,0]) == 0
    assert fsa([1,0,2]) > 0
    assert 371 < fsa.pathsum() < 372


# every symbols have two path, one project to the

class SSM(torch.nn.Module):
    def __init__(self, A, B, C, init=None, phi=None, pi=None):
        """
        A: state matrix (K x K): determines contribution of state at previous
           time to state at current time
        B: input matrix (K x S): determines contribution of input at current
           time to state at current time
        C: output matrix (S x K): maps states to output space

        Optional:
        phi: matrix (Y x S): maps segments to vector representation
        pi: vector (S): projection vector for tier. Segments s with pi[s]=0 are ignored.
        init: initialization vector (K); if unspecified, [0, 0, 0, ..., 0, 1] by default.
        """
        super().__init__()
        self.A = A
        self.B = B
        self.C = C

        # Confirm that A,B,C have correct dimensions
        X, X2 = self.A.shape
        X3, U = self.B.shape
        Y, X4 = self.C.shape
        assert X == X2 == X3 == X4

        self.dtype = self.A.dtype

        self.isemiring = BooleanSemiring if self.dtype is torch.bool else RealSemiring
        self.semiring = LogspaceSemiring # output will always be logits

        # NOTE: SSM not set up to run in Logspace semiring
        self.phi = torch.eye(U, dtype=self.dtype, device=DEVICE) if phi is None else phi # default to identity matrix
        self.init = torch.eye(X, dtype=self.dtype, device=DEVICE)[0] if init is None else init # default to [1, 0, 0, 0, ...]

        # The projection matrix pi is a function from input feature i to state feature j.
        # It says, for input feature i, whether state feature j should be sensitive to it.
        # By default, all state features are sensitive to all input features, yielding a standard LTI SSM.
        if pi is None:
            self.pi = torch.ones(U, X, dtype=self.dtype, device=DEVICE) # default [[1, 1, ...], ...].T
        else:
            self.pi = pi
            assert self.pi.shape[0] == U
            assert self.pi.shape[1] == X

    def forward(self, input, debug=False):
        T = len(input)
        u = self.phi[input]
        proj = self.isemiring.mm(u, self.pi) # torch.einsum("ux,tu->tx", self.pi, u)
        x = [torch.zeros(self.A.shape[0], dtype=self.dtype, device=DEVICE) for _ in range(T+1)]
        x[0] = self.init
        for t in range(T):
            update = self.isemiring.mv(self.A, x[t]) + self.isemiring.mv(self.B, u[t])
            x[t+1] = self.isemiring.complement(proj[t])*x[t] + proj[t]*update
            if debug:
                breakpoint()
        x = torch.stack(x)
        y = torch.einsum("yx,tx->ty", (self.C * self.pi), x[:-1].float())
        return SSMOutput(u, proj, x, y)

    def forward_incremental(self, input, debug=False):
        result = self.forward(input, debug=debug)
        return result.y

# Classes for trainable phonotactics models

class PhonotacticsModel(torch.nn.Module):
    def __add__(self, other):
        weights = torch.nn.Parameter(1/DEFAULT_INIT_TEMPERATURE * torch.randn(2))
        return MixturePhonotacticsModel(
            self, other,
            weights=weights,
            logspace_weights=True
        )

    def __mul__(self, other):
        return ProductPhonotacticsModel(self, other)

    def train(self,
              batches: Iterable[Iterable[Tuple[int, Sequence[int]]]], # iterable of batches of sequences of ints
              report_every: int=1000,
              debug: bool=False,
              reporting_window_size: int=100,
              natural_gradient: bool=False, # broken
              checkpoint_prefix: Optional[str]=None,
              eval_fn: Optional[Callable[[torch.nn.Module], Any]]=None,
              hyperparams_to_report: Optional[Dict]=None,
              **kwds):
        if natural_gradient:
            opt = naturalgradient.NaturalGradientDescent(params=self.parameters(), model=self, **kwds)
        else:
            opt = torch.optim.Adam(params=self.parameters(), **kwds)            
        
        reporting_window = deque(maxlen=reporting_window_size)
        diagnostics = []
        writer = csv.writer(sys.stdout)
        first_time = True
        for i, (epoch, batch) in enumerate(batches):
            if i % report_every == 0:
                diagnostic = {
                    'step': i,
                    'epoch': epoch,
                }
                with torch.no_grad():
                    diagnostic |= eval_fn(self)
                if hyperparams_to_report:
                    diagnostic |= hyperparams_to_report
                if checkpoint_prefix is not None:
                    filename = checkpoint_prefix + "_%d.pt" % i
                    with open(filename, 'wb') as outfile:
                        torch.save(self, outfile)
            opt.zero_grad()
            loss = -self.log_likelihood(batch, debug=debug).mean()
            loss.backward()
            opt.step()
            reporting_window.append(loss.detach())
            if i % report_every == 0:
                mean_loss = torch.stack(list(reporting_window)).mean().item()
                diagnostic['mean_loss'] = mean_loss
                if first_time:
                    writer.writerow(list(diagnostic.keys()))
                    first_time = False
                writer.writerow(list(diagnostic.values()))
                diagnostics.append(diagnostic)                
        return diagnostics

class ProductPhonotacticsModel(PhonotacticsModel):
    def __init__(self, *models, weights=None, logspace_weights=False):
        super(PhonotacticsModel, self).__init__()
        self.models = torch.nn.ModuleList(list(models))
        if weights is None:
            self.weights = torch.ones(len(self.models), device=DEVICE)
        elif logspace_weights:
            self.weights = weights.exp()
        else:
            self.weights = weights

    def log_likelihood(self, xs: Iterable[Sequence[int]], debug: bool=False):
        # Always locally normalized / delimited! Does not use final probs.
        automata = [model.automaton() for model in self.models]
        def gen():
            for x in xs:
                logits = torch.stack([
                    a.semiring.to_log(a.forward_incremental(x, debug=debug))
                    for a in automata
                ]) # MTS
                combo_logits = (self.weights[:, None, None] * logits).sum(0)
                yield combo_logits.log_softmax(-1).gather(-1, x.unsqueeze(-1)).sum()
        result = torch.stack(list(gen()))
        return result

class MixturePhonotacticsModel(PhonotacticsModel):
    def __init__(self, *models, weights=None, logspace_weights=False):
        super(PhonotacticsModel, self).__init__()
        self.models = torch.nn.ModuleList(list(models))
        if weights is None:
            self.weights = torch.zeros(len(self.models), device=DEVICE)
            self.logspace_weights = True
        else:
            self.weights = weights
            self.logspace_weights = logspace_weights

    def log_likelihood(self, xs: Iterable[Sequence[int]], debug: bool=False):
        ys = torch.stack([model.log_likelihood(x) for model in self.models])
        if self.logspace_weights: 
            weights = self.weights.log_softmax(-1)
        else: 
            weights = self.weights.log().log_softmax(-1)
        y = (weights[:, None] + ys).logsumexp(0)
        return y


class FSAPhonotacticsModel(PhonotacticsModel):
    """ FSA Phonotactics model parameterized by log-space weights. """
    
    def __init__(self, A, init=None, final=None, semiring=RealSemiring):
        super().__init__()
        # the logspace weights here will be transformed into a WFSA via
        # self.A and self.final -> self.A_and_final_positive -> self.A_and_final_normalized -> self.fsa
        self.A = A # unnormalized weights
        Q, S, Q2 = self.A.shape
        assert Q == Q2

        self.semiring = semiring        

        self.final = final # might be None
        if init is None:
            self.init = self.semiring.from_exp(torch.nn.functional.one_hot(torch.tensor(0), Q)).to(DEVICE)
        else:
            self.init = init

    def A_and_final_positive(self):
        A_positive = self.semiring.from_log(self.A)
        if self.final is None:
            return A_positive, None
        else:
            final_positive = self.semiring.from_log(self.final)
            return A_positive, final_positive

    def forward(self, xss, debug=False):
        return self.log_likelihood(xss, debug=debug)

    def automaton(self, semiring=None):
        return self.fsa(semiring=semiring)

    def fsa(self, semiring=None) -> WFSA:
        A, final = self.A_and_final_normalized()
        return WFSA(
            A,
            init=self.init,
            final=final,
            semiring=self.semiring if semiring is None else semiring
        )

    @classmethod
    def initialize(
            cls,
            X,
            S,
            learn_final=False,
            requires_grad=True,
            semiring=RealSemiring,
            init_T=DEFAULT_INIT_TEMPERATURE):
        A = torch.nn.Parameter((1/init_T)*torch.randn(X, S, X), requires_grad=requires_grad)
        if learn_final:
            final = torch.nn.Parameter((1/init_T)*torch.randn(X), requires_grad=requires_grad)
            model = cls(A, final=final, semiring=semiring)
        else:
            model = cls(A, semiring=semiring)
        return model.to(DEVICE)

class OverparameterizedFSAPhonotacticsModel(FSAPhonotacticsModel):
    def __init__(self, B, C, init=None, final=None, semiring=RealSemiring):
        super(FSAPhonotacticsModel, self).__init__()
        self.B = B # unnormalized weights
        self.C = C
        BQ, BQ2 = self.B.shape
        CQ, S, CQ2 = self.C.shape
        assert BQ == CQ2
        assert BQ2 == CQ

        self.semiring = semiring
        self.final = final # might be None        
        if init is None:
            self.init = self.semiring.from_exp(torch.nn.functional.one_hot(torch.tensor([0]), CQ2)).to(DEVICE)
        else:
            self.init = init

    @classmethod
    def initialize(
            cls,
            X,
            H,
            S,
            learn_final=False,
            requires_grad=True,
            semiring=RealSemiring,
            init_T=DEFAULT_INIT_TEMPERATURE):
        B = torch.nn.Parameter((1/init_T)*torch.randn(X, H), requires_grad=requires_grad)
        C = torch.nn.Parameter((1/init_T)*torch.randn(H, S, X), requires_grad=requires_grad)
        if learn_final:
            final = torch.nn.Parameter((1/init_T)*torch.randn(X), requires_grad=requires_grad)
            model = cls(B, C, final=final, semiring=semiring)
        else:
            model = cls(B, C, semiring=semiring)
        return model.to(DEVICE)

    def A_and_final_positive(self):
        BQ, BQ2 = self.B.shape
        CQ, S, CQ2 = self.C.shape        
        C = self.C.reshape(CQ, S*CQ2)
        A = self.semiring.mm(self.B, C)
        A_positive = self.semiring.from_log(A) # Q1 S Q2
        final = None if self.final is None else self.semiring.from_log(self.final)
        return A_positive.reshape(BQ, S, CQ2), final


def soft_ceiling(x, k, beta=1):
    return k - torch.nn.functional.softplus(k - x, beta=beta)

def spectral_radius(A):
    return torch.linalg.eigvals(A).abs().max()

class GloballyNormalized:
    def A_and_final_normalized(self):
        return self.A_and_final_positive()

    def log_likelihood(self, xs: Iterable[Sequence[int]], debug: Optional[bool]=False):
        wfsa = self.fsa()
        y = torch.stack([wfsa(x, debug=debug) for x in xs])
        Z = self.fsa(semiring=RealSemiring).pathsum()
        return self.semiring.to_log(y) - logZ

    
class AdjustedNormalized(GloballyNormalized):
    def A_and_final_normalized(self):
        A_positive, final_positive = self.A_and_final_positive()
        s = spectral_radius(A_positive.sum(-2) + final_positive[:, None])
        adjustment = soft_ceiling(s, 1) / s
        A_normalized = adjustment * A_positive
        final_normalized = adjustment * final_positive
        return self.semiring.from_exp(A_normalized), self.semiring.from_exp(final_normalized)
    
        
class LocallyNormalized: # NOTE: PFSA models can have a halting probability, SSM models don't.
    def log_likelihood(self, xs: Iterable[Sequence[int]], debug: Optional[bool]=False):
        pfsa = self.fsa()
        y = torch.stack([pfsa(x, debug=debug) for x in xs])
        return self.semiring.to_log(y)

    def A_and_final_normalized(self):
        A_positive = self.semiring.from_log(self.A)
        final_positive = self.semiring.from_log(self.final)
        Z = self.semiring.add(self.semiring.sum(A_positive, dim=(-1, -2)), final_positive)
        A_normalized = self.semiring.div(A_positive, Z[:, None, None])
        final_normalized = self.semiring.div(final_positive, Z)
        return A_normalized, final_normalized

    def fsa_fisher(self):
        # Fisher Information Metric F(A) as a function of probabilities A_ijk = p(s_j, q_k | q_i).
        # Then for underlying parameters \theta, F(\theta) = J F(A) J^T, where J = dA/d\theta.
        pass
        


class LocallyNormalizedWithEOS(LocallyNormalized):
    def A_and_final_normalized(self):
        A_positive, _ = self.A_and_final_positive()
        Z = self.semiring.sum(A_positive, dim=(-1, -2))
        A_normalized = self.semiring.div(A_positive, Z[:, None, None])
        return A_normalized, None
    
    def fsa(self, semiring=None) -> WFSA:
        """ WFSA with an explicit final sink state after eos """
        A_normalized, _ = self.A_and_final_normalized()
        Q, S, R = A_normalized.shape
        # remove probability mass associated with eos and ensuing states
        
        # sink state is characterized by
        # p(sink | :, :) = 0        
        # p(sink | eos, :) = 1
        # thus
        # p(sink, : | :) = 0
        # p(sink, eos | :) = p(eos | :) * 1
        A_delimited = self.semiring.from_exp(torch.zeros(Q+1, S, R+1, device=DEVICE))
        A_delimited[:-1, :, :-1] = A_normalized
        A_delimited[:, 0, -1] = self.semiring.sum(A_delimited[:, 0, :], -1) # redirect all probability mass into the sink
        A_delimited[:, 0, :-1] = self.semiring.zero # after eos, never go into any state except the sink

        final = self.semiring.from_exp(torch.zeros(Q+1, device=DEVICE))
        final[-1] = self.semiring.one
        init = torch.full_like(final, self.semiring.zero)
        init[:-1] = self.init
        return WFSA(
            A_delimited,
            init=init,
            final=final,
            semiring=self.semiring if semiring is None else semiring,
        )
    

class PFSAPhonotacticsModel(LocallyNormalizedWithEOS, FSAPhonotacticsModel):
    pass

class OverparameterizedPFSAPhonotacticsModel(LocallyNormalizedWithEOS, OverparameterizedFSAPhonotacticsModel):
    pass

class WFSAPhonotacticsModel(AdjustedNormalized, FSAPhonotacticsModel):
    pass

class pTSL(PFSAPhonotacticsModel):
    def __init__(self, Eb, pi, final=None, bias=False, semiring=RealSemiring):
        super(PhonotacticsModel, self).__init__()
        self.Eb = Eb # shape QS
        self.pi = pi # shape S
        self.bias = bias
        Q, S = self.E.shape
        assert S == self.pi.shape[-1]

        self.semiring = semiring
        self.init = self.semiring.from_exp(torch.nn.functional.one_hot(torch.tensor(0), Q)).to(DEVICE)
        self.final = final

        self.T_on_tier = self.semiring.from_exp(torch.cat([torch.zeros(1, S), torch.eye(S)]).to(DEVICE)).T # shape 1SQ, saving symbol as state
        self.T_not_on_tier = self.semiring.from_exp(torch.eye(Q, device=DEVICE))[:, None, :] # shape Q1Q, preserving state

    @property
    def E(self):
        if self.bias:
            return self.Eb[:, 0].unsqueeze(-1) + self.Eb[:, 1:]
        else:
            return self.Eb
    
    @classmethod
    def initialize(cls,
                   S,
                   pi=None,
                   bias=False,
                   learn_final=False,
                   semiring=RealSemiring,
                   requires_grad=True,
                   init_T=DEFAULT_INIT_TEMPERATURE):
        if pi is None:
            pi = torch.nn.Parameter((1/init_T)*torch.randn(S), requires_grad=requires_grad)
        E = torch.nn.Parameter((1/init_T)*torch.randn(S+1, S+bias), requires_grad=requires_grad)
        if learn_final:
            final = torch.nn.Parameter((1/init_T)*torch.randn(S+1), requires_grad=requires_grad)
            model = cls(E, pi, bias=bias, final=final, semiring=semiring)
        else:
            model = cls(E, pi, bias=bias, semiring=semiring)
        return model.to(DEVICE)

    def A_and_final_positive(self):
        # In real semiring:
        # A = proj * self.E.exp()[:, :, None] * self.T_on_tier + (1-proj) * self.T_not_on_tier
        proj = self.semiring.logistic(self.pi)[None, :, None]
        not_proj = self.semiring.logistic(-self.pi)[None, :, None]
        E = self.semiring.from_log(self.E)[:, :, None] 
        on_tier = self.semiring.mul(E, self.T_on_tier)
        one = self.semiring.mul(proj, on_tier)
        two = self.semiring.mul(not_proj, self.T_not_on_tier)
        A = self.semiring.add(one, two) 
        return A, None

class SSMPhonotacticsModel(PhonotacticsModel):
    def __init__(self, A_params, B_params, Cb_params, pi_params=None, init=None, bias=False):
        super().__init__()
        self.A_params = A_params
        self.B_params = B_params
        self.Cb_params = Cb_params
        self.init = init
        self.pi_params = pi_params
        self.bias = bias

    @property
    def A(self):
        return self.A_params

    @property
    def B(self):
        return self.B_params

    @property
    def C(self):
        if self.bias:
            C = self.Cb_params[0, :] + self.Cb_params[1:, :]
        else:
            C = self.Cb_params
        return C

    @property
    def pi(self):
        return self.pi_params

    @classmethod
    def initialize(
            cls,
            X,
            S,
            requires_grad=True,
            bias=False,
            init_T_A=DEFAULT_INIT_TEMPERATURE,
            init_T_B=DEFAULT_INIT_TEMPERATURE,
            init_T_C=DEFAULT_INIT_TEMPERATURE):
        A = torch.nn.Parameter((1/init_T_A)*torch.randn(X, X), requires_grad=requires_grad)
        B = torch.nn.Parameter((1/init_T_B)*torch.randn(X, S), requires_grad=requires_grad)
        C = torch.nn.Parameter((1/init_T_C)*torch.randn(S+bias, X), requires_grad=requires_grad)
        return cls(A, B, C, bias=bias).to(DEVICE)

    def log_likelihood(self, xs: Iterable[torch.LongTensor], debug: bool=False):
        ssm = self.ssm()
        def gen():
            for x in xs:
                # for sequence x1 x2 x3, get the vector of logits of x_t | x_{<t}
                logprobs = ssm(x, debug=debug).y
                # normalize locally -- cannot in general normalize C in advance
                yield logprobs.log_softmax(-1).gather(-1, x.unsqueeze(-1)).sum()
        return torch.stack(list(gen()))

    forward = log_likelihood

    def automaton(self, semiring=None):
        return self.ssm()

    def ssm(self) -> SSM:
        return SSM(self.A, self.B, self.C, init=self.init, pi=self.pi)

    def __mul__(self, other):
        if isinstance(other, SSMPhonotacticsModel):
            return ProductSSMModel(self, other)
        else:
            return ProductPhonotacticsModel(self, other)

class DiagonalSSMPhonotacticsModel(SSMPhonotacticsModel):
    @classmethod
    def initialize(
            cls,
            X,
            S,
            requires_grad=True,
            A_diag=None,
            B=None,
            Cb=None,
            bias=False,
            init_T_A=DEFAULT_INIT_TEMPERATURE,
            init_T_B=DEFAULT_INIT_TEMPERATURE,
            init_T_C=DEFAULT_INIT_TEMPERATURE):
        if A_diag is None:
            A_diag = torch.nn.Parameter((1/init_T_A)*torch.randn(X), requires_grad=requires_grad)
            
        if B is None:
            B = torch.nn.Parameter((1/init_T_B)*torch.randn(X, S), requires_grad=requires_grad)

        if Cb is None:
            Cb = torch.nn.Parameter((1/init_T_C)*torch.randn(S+bias, X), requires_grad=requires_grad)

        return cls(A_diag, B, Cb, bias=bias).to(DEVICE)

    @property
    def A(self):
        return torch.diag(self.A_params)

    
class SquashedDiagonalSSMPhonotacticsModel(DiagonalSSMPhonotacticsModel):
    @property
    def A(self):
        return torch.diag(torch.sigmoid(self.A_params))

    @property
    def B(self):
        return torch.tanh(self.B_params)
    

class ProductSSMModel(SSMPhonotacticsModel):
    def __init__(self, *models):
        super(SSMPhonotacticsModel, self).__init__()
        self.models = torch.nn.ModuleList(list(models))

    def ssm(self) -> SSM:
        ssms = [m.ssm() for m in self.models]
        return SSM(
            A=torch.block_diag(*[m.A for m in ssms]),
            B=torch.cat([m.B for m in ssms]),
            C=torch.cat([m.C for m in ssms], dim=-1),
            init=torch.cat([m.init for m in ssms]),
            pi=torch.cat([m.pi for m in ssms], dim=-1),
        )

class Factor2(DiagonalSSMPhonotacticsModel):
    """ Phonotactics model whose probabilies are determined by factors of 2 segments """
    @classmethod
    def from_factors(cls, factors, projection=None, bias=False):
        S, X = factors.shape
        A, B, init = cls.init_matrices(X)
        return cls(A, B, factors, init=init, pi_params=projection, bias=bias)

    @classmethod
    def init_matrices(cls, d, bias=False):
        raise NotImplementedError

    @classmethod
    def initialize(cls, S, bias=False, projection=None, requires_grad: bool=True, init_T=DEFAULT_INIT_TEMPERATURE):
        factors = torch.nn.Parameter((1/init_T)*torch.randn(S+bias, S+1), requires_grad=requires_grad)
        return cls.from_factors(factors, projection=projection, bias=bias).to(DEVICE)


class SL2(Factor2):
    @classmethod
    def init_matrices(cls, d):
        A_diag = torch.zeros(d, dtype=bool, device=DEVICE)  # add a bias dimension
        B = torch.eye(d, dtype=bool, device=DEVICE)[:, 1:] # X x S
        init = torch.eye(d, dtype=bool, device=DEVICE)[0]
        return A_diag, B, init

class SP2(Factor2):
    @classmethod
    def init_matrices(cls, d):
        A_diag = torch.ones(d, dtype=bool, device=DEVICE)
        B = torch.eye(d, dtype=bool, device=DEVICE)[:, 1:]
        init = torch.eye(d, dtype=bool, device=DEVICE)[0]
        return A_diag, B, init

class QuasiSP2(Factor2):
    @classmethod
    def init_matrices(cls, d):
        A_diag = torch.ones(d, device=DEVICE)
        B = torch.eye(d, device=DEVICE)[:, 1:]
        init = torch.eye(d, device=DEVICE)[0]
        return A_diag, B, init

class TierBased(Factor2):
    """ Assume projection is a function from input features to a single number for all state features,
    that is, we can say that a segment is or is not projected. """

    @property
    def pi(self):
        return self.pi.unsqueeze(-1).expand(*self.C.shape) 

    
class SoftTierBased(Factor2):
    @classmethod
    def initialize(
            cls,
            S,
            pi=None,
            bias=False,
            requires_grad: bool=True,
            init_T=DEFAULT_INIT_TEMPERATURE,
            init_T_projection=DEFAULT_INIT_TEMPERATURE):
        factors = torch.nn.Parameter((1/init_T)*torch.randn(S, S+1), requires_grad=requires_grad)
        if pi is None:
            projection = torch.nn.Parameter((1/init_T_projection)*torch.randn(S), requires_grad=requires_grad)
        else:
            projection = pi
        return cls.from_factors(factors, projection, bias=bias).to(DEVICE)

    @property
    def pi(self):
        return self.pi.sigmoid().unsqueeze(-1).expand(*self.C.shape)


class ProbabilisticTierBased(SoftTierBased):

    @property
    def C(self):
        if self.bias:
            C = self.Cb_params[0, :] + self.Cb_params[1:, :]
        else:
            C = self.Cb_params
        return C.exp()
    
    def log_likelihood(self, xs: Iterable[torch.LongTensor], debug: Optional[bool]=False):
        ssm = self.ssm()
        def gen():
            for x in xs:
                weights = ssm(x).y + (1-ssm.pi)[:, None] # hack
                # Assume the weights are already positive, so we only need to normalize, not softmax
                lnZ = weights.sum(-1).log() # shape T
                relevant = weights.gather(-1, x.unsqueeze(-1)).log()
                yield relevant - lnZ
        return torch.stack(list(gen()))

    forward = log_likelihood

class TSL2(SL2, TierBased):
    pass

class TSP2(SP2, TierBased):
    pass

class SoftTSL2(SL2, SoftTierBased):
    pass


# Data handling

def whole_dataset(data: Iterable, num_epochs: int=1) -> Iterator[Sequence]:
    return minibatches(data, len(data), num_epochs=num_epochs)

def single_datapoints(data: Iterable, num_epochs: int=1) -> Iterator[Sequence]:
    return minibatches(data, 1, num_epochs=num_epochs)

def batch(iterable: Sequence, n: int=1) -> Iterator[Sequence]:
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i : min(i+n, l)]

def minibatches(data: Iterable,
                batch_size: int=1,
                num_epochs: int=1):
    """
    Generate a stream of data in minibatches of size batch_size.
    Go through the data num_epochs times, each time in a random order.
    If num_epochs not specified, returns an infinite stream of batches.
    """
    data = list(data)
    for epoch in range(num_epochs):
        random.shuffle(data)
        for datapoint in batch(data, batch_size):
            yield epoch, datapoint

# Example SSMs

def anbn_ssm():
    """ SSM for a^n b^n. a = 1, b = 2, halt = 0.
    For every length N, the highest-probability string is a^{(N-1)/2} b^{(N-1)/2} #
    """
    A = torch.diag(torch.Tensor([
        1, # dimension for count of a's
        1, # dimension for whether we have seen b
    ]))
    B = torch.Tensor([
        [0, 1, -1],  # increment for a, decrement for b
        [0, 0, 10],   # increase for each b
    ])
    C = torch.Tensor([
        [-10, 0], # halt -- only if no outstanding a's.
        [0, -10], # a -- not allowed if b seen.
        [0, 0],
    ])
    return SSM(A, B, C, init=torch.zeros(2))


# Generating random samples from languages of interest

def random_star_ab(S=3, T=4):
    while True:
        s = [random.choice(range(3)) for _ in range(T)]
        if not any(x == 0 and y == 1 for x, y in zip(s, s[1:])):
             return s

def random_and():
    V1 = random.choice([0,1])
    V2 = random.choice([0,1])
    V3 = 1 if V1 and V2 else random.choice([0,1])
    return [V1, V2, V3]

def random_no_axb(n=4, neg=False):
    # no sequence 20*3
    # 0 is not projected so should be ignored.
    # 1 is projected so it should not be ignored: 213 is ok.
    while True:
        sequence = [random.choice(range(4)) for _ in range(n)]
        tier = [x for x in sequence if x != 0]
        if any(x == 2 and y == 3 for x, y in pairs(tier)) == neg:
            return sequence

def random_no_ab(n=4, neg=False):
    # No substring 23
    while True:
        sequence = [random.choice(range(4)) for _ in range(n)]
        if any(x == 2 and y == 3 for x, y in pairs(sequence)) == neg:
            return sequence

def random_no_ab_subsequence(n=4, neg=False):
    # No subsequence 23
    while True:
        sequence = [random.choice(range(4)) for _ in range(n)]
        bad = False
        for i, sym1 in enumerate(sequence):
            if sym1 == 2:
                for j, sym2 in enumerate(sequence[i+1:]):
                    if sym2 == 3:
                        bad = True
        if not neg and not bad:
            return sequence
        elif neg and bad:
            return sequence

def pairs(xs): # contiguous substrings of length 2
    return zip(xs, xs[1:])

def random_two_tiers(n=6):
    # two independent tiers, {012} and {345}.
    # Prohibit *01 and *34 on each tier.
    p1 = {0,1,2}
    p2 = {3,4,5}
    while True:
        sequence = [random.choice(range(2*3)) for _ in range(n)]
        tier1 = [x for x in sequence if x in p1]
        tier2 = [x for x in sequence if x in p2]
        if (not any(x == 0 and y == 1 for x, y in pairs(tier1))
            and not any(x == 3 and y == 4 for x, y in pairs(tier2))):
            return sequence

def random_tiptup():
    C1 = random.choice([0,1])
    C2 = random.choice([0,1])
    V = 2 + C1 ^ C2
    return [C1, V, C2]

def random_xor():
    two = [random.choice(range(2)) for _ in range(2)]
    return two + [two[0] ^ two[1]]

def random_anbn(p_halt=1/2, start=1):
    T = start
    while True:
        if random.random() < p_halt:
            break
        else:
            T += 1
    return [1]*T + [2]*T + [0]


# Evaluation functions

def evaluate_and(model):
    good = [
        [0,0,0],
        [0,0,1],
        [0,1,0],
        [0,1,1],
        [1,0,0],
        [1,0,1],
        [1,1,1],
    ]
    bad = [
        [1,1,0],
    ]
    return evaluate_model_unpaired(model, good, bad)

def evaluate_tiptup(model):
    good = [
        [0,2,0],
        [1,2,1],
        [0,3,1],
        [1,3,0],
    ]
    bad = [
        [1,2,0],
        [0,2,1],
        [1,3,1],
        [0,3,0]
    ]
    return evaluate_model_unpaired(model, good, bad)

def evaluate_no_axb(num_epochs=20,
                    batch_size=5,
                    n=4,
                    model_type=TSL2,
                    force_pi=None,
                    num_samples=1000,
                    num_test=100,
                    **kwds):
    """ Evaluation for 2-TSL """
    dataset = list(map(torch.LongTensor, [random_no_axb(n=n) for _ in range(num_samples)]))
    # the first two of these comparisons will be exactly zero for SL
    # the third comparison will be exactly zero for SL and SP

    good = list(map(torch.LongTensor, [ # no 23 on the tier {1, 2, 3}
         [0,2,0,1,0,3], # matched on SL factors
         [0,0,2,0,0,1,0,0,3], # matched on SL factors
         [0,2,0,1,0,2,0,1,0,3], # matched on (boolean) SP factors: 00,01,02,03,10,11,12,13,20,21,22,23
         [1,1,3,0,2,0],
         [3,3,2,1,3,0],
    ]))

    bad = list(map(torch.LongTensor, [
        [0,1,0,2,0,3],
        [0,0,1,0,0,2,0,0,3],
        [0,2,0,1,0,1,0,2,0,3], # same SP factors as the SP example above;
        [1,1,2,0,3,0],
        [3,3,2,0,3,0],
    ]))
    #good = [random_no_axb(n=n+1) for _ in range(num_test)]
    #bad = [random_no_axb(n=n+1, neg=True) for _ in range(num_test)]

    if issubclass(model_type, TierBased): # fixed tier -- the right projection function is [0,1,1,1].
        model = model_type.initialize(4, torch.Tensor([0,1,1,1])) 
    elif force_pi is not None:
        model = model_type.initialize(4, pi=force_pi)
    else:
        model = model_type.initialize(4)

    eval_fn = lambda m: {'good_bad_diff': evaluate_model_paired(m, good, bad)['diff'].sum()}
    data = minibatches(dataset, batch_size, num_epochs=num_epochs)
    model.train(data, eval_fn=eval_fn, **kwds)
    return evaluate_model_paired(model, good, bad), model

def evaluate_no_ab(num_epochs=20, batch_size=5, n=4, model_type=SL2, num_samples=1000, num_test=100, **kwds): # SL2 dataset
    dataset = [random_no_ab(n=n) for _ in range(num_samples)]

    # good = [
    #     [0, 2, 1, 3, 2, 2], # matched on 2-SP factors
    #     [2, 0, 3, 1, 3, 2], # TSL model will fail
    #     [3, 2, 1, 0, 1, 3],
    #     [1, 3, 2, 0, 3, 1], # TSL model will fail
    #     [0, 1, 2, 1, 3, 2],
    # ]
    # bad = [
    #     [0, 2, 1, 2, 3, 2], # matched on 2-SP factors
    #     [2, 3, 0, 1, 3, 2],
    #     [3, 2, 1, 0, 2, 3],
    #     [1, 3, 2, 3, 0, 1],
    #     [0, 1, 2, 3, 1, 2],
    # ]
    good = [random_no_ab(n=n+1) for _ in range(num_test)]
    bad = [random_no_ab(n=n+1, neg=True) for _ in range(num_test)]

    if issubclass(model_type, TierBased):
        model = model_type.initialize(4, torch.Tensor([0,1,1,1]))
    else:
        model = model_type.initialize(4)

    data = minibatches(dataset, batch_size, num_epochs=num_epochs)
    model.train(data, **kwds)
    return evaluate_model_unpaired(model, good, bad)

    # 2-SP language *a-a
    # 2-TSL language: *a-a, Tier = {a, b, c, d}

    # For a string like aba
    # - this is prohibited by SP but allowed by TSL
    # but for Tier = {a} then no

def evaluate_no_ab_subsequence(num_epochs=20, batch_size=5, n=4, model_type=SP2, num_samples=1000, num_test=100, **kwds):
    """ Evaluation for 2-SP """
    dataset = [random_no_ab_subsequence(n=n) for _ in range(num_samples)]
    # the first two of these comparisons will be exactly zero for SL
    # the third comparison will be exactly zero for SL and SP
    # good = [ # no 23 on the tier {1, 2, 3}
    #     [0,2,0,1,0,3], # matched on SL factors
    #     [0,0,2,0,0,1,0,0,3], # matched on SL factors
    #     [0,2,0,1,0,2,0,1,0,3], # matched on (boolean) SP factors: 00,01,02,03,10,11,12,13,20,21,22,23
    #     [1,1,3,0,2,0],
    #     [3,3,2,1,3,0],
    # ]

    # bad = [
    #     [0,1,0,2,0,3],
    #     [0,0,1,0,0,2,0,0,3],
    #     [0,2,0,1,0,1,0,2,0,3], # same SP factors as the SP example above;
    #     [1,1,2,0,3,0],
    #     [3,3,2,0,3,0],
    # ]
    good = [random_no_ab_subsequence(n=n+1) for _ in range(num_test)]
    bad = [random_no_ab_subsequence(n=n+1, neg=True) for _ in range(num_test)]

    if issubclass(model_type, TierBased):
        model = model_type.initialize(4, torch.Tensor([0,1,1,1]))
    else:
        model = model_type.initialize(4)

    data = minibatches(dataset, batch_size, num_epochs=num_epochs)
    model.train(data, **kwds)
    return evaluate_model_unpaired(model, good, bad)

def evaluate_model_paired(model: SSMPhonotacticsModel, good_strings, bad_strings):
    def gen():
        for good, bad in zip(good_strings, bad_strings):
            good = torch.LongTensor(good)
            bad = torch.LongTensor(bad)
            yield good, bad, model.log_likelihood([good]).mean().item(), model.log_likelihood([bad]).mean().item()
    df = pd.DataFrame(list(gen()))
    df.columns = ['good', 'bad', 'good_score', 'bad_score']
    df['diff'] = df['good_score'] - df['bad_score']
    return df

def evaluate_model_unpaired(model, good_strings, bad_strings):
    """ Do a pairwise comparison of log likelihood for all good and bad strings. """
    def gen():
        for good in good_strings:
            for bad in bad_strings:
                yield good, bad, model.log_likelihood([good]).mean().item(), model.log_likelihood([bad]).mean().item()
    df = pd.DataFrame(list(gen()))
    df.columns = ['good', 'bad', 'good_score', 'bad_score']
    df['diff'] = df['good_score'] - df['bad_score']
    return df

def evaluate_model_simple(model, good_strings, bad_strings):
    df_list = []
    for good in good_strings:
        df_list.append([good, 'good', model.log_likelihood([good]).mean().item()])
    for bad in bad_strings:
        df_list.append([bad, 'bad', model.log_likelihood([bad]).mean().item()])
    df = pd.DataFrame(df_list)
    df.columns = ['string', 'grammatical', 'score']
    return df

if __name__ == '__main__':
    import nose
    nose.runmodule()
