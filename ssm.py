import sys
import csv
import random
from typing import *
import operator
import functools
import itertools
from collections import namedtuple, deque

# import tqdm
import torch
import pandas as pd
import numpy as np
import torch_semiring_einsum as tse
# import matplotlib.pyplot as plt

INF = float('inf')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# DEVICE = torch.device("mps") if torch.backends.mps.is_available() else 'cpu'

# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# print("Using device:", device)

#DEVICE = 'cpu'

DEFAULT_INIT_TEMPERATURE = 100
EPSILON = 10 ** -5

#torch.autograd.set_detect_anomaly(True)

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
    sum = torch.sum
    prod = torch.prod
    div = operator.truediv
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
    add = torch.logaddexp
    mul = operator.add
    sum = torch.logsumexp
    prod = torch.sum
    div = operator.sub
    from_exp = torch.log
    from_log = lambda x:x
    to_exp = torch.exp
    to_log = lambda x:x
    logistic = torch.nn.functional.logsigmoid

    mv_eq = tse.compile_equation("ij,j->i")
    bmv_eq = tse.compile_equation("bij,j->bi")
    mm_eq = tse.compile_equation("ij,jk->ik")

    @classmethod
    def mv(cls, A, x):
        if A.ndim == 2:
            return tse.log_einsum(cls.mv_eq, A, x)
        elif A.ndim == 3:
            return tse.log_einsum(cls.bmv_eq, A, x)
        else:
            raise ValueError("Bad dimensions: A=%s, x=%s" % (str(A.shape), str(x.shape)))

    @classmethod
    def mm(cls, A, B):
        return tse.log_einsum(cls.mm_eq, A, B)

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
    def __init__(self, A, init=None, final=None, phi=None, semiring=None):
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
            
        self.phi = torch.eye(Y1, dtype=self.dtype, device=DEVICE) if phi is None else phi # default to identity matrix
        self.init = torch.eye(X1, dtype=self.dtype, device=DEVICE)[0] if init is None else init # default to [1, 0, 0, 0, ...], enforcing state 0 = initial state.
        self.final = self.semiring.from_exp(torch.ones(X1, dtype=self.dtype, device=DEVICE)) if final is None else final # default to [..., 1, 1, 1], meaning all states can be equally final

    def forward(self, input, init=None, final=None, debug=False):
        init = self.semiring.from_exp(self.init if init is None else init)
        final = self.final if final is None else final
        A_transposed = self.A.transpose(-1, -2)        
        u = self.semiring.from_exp(self.phi[input])        
        x = init
        for u_t in u:
            #x = self.semiring.einsum("qsr,s,q->r", self.A, u_t, x)
            A_t = self.semiring.mv(A_transposed, u_t)
            x = self.semiring.mv(A_t.T, x)
        y = self.semiring.vv(x, final)
        if debug:
            breakpoint()
        return y

    def transition_closure(self):
        # (I - A)^{-1} via solve; no spectral radius check in the hot path
        A = self.A.sum(-2)
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        return torch.linalg.solve(I - A, I)  # faster/more stable than inv

    def pathsum(self, init=None, final=None):
        init = self.init if init is None else init
        final = self.final if final is None else final
        A_star = self.transition_closure()
        return self.semiring.vv(init, self.semiring.mv(A_star, final))

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

        self.semiring = BooleanSemiring if self.dtype is torch.bool else RealSemiring

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

    def forward(self, input, init=None, debug=False):
        T = len(input)
        u = self.phi[input]
        proj = self.semiring.mm(u, self.pi) # torch.einsum("ux,tu->tx", self.pi, u)
        x = [torch.zeros(self.A.shape[0], dtype=self.dtype, device=DEVICE) for _ in range(T+1)]
        x[0] = self.init if init is None else init
        for t in range(T):
            update = self.semiring.mv(self.A, x[t]) + self.semiring.mv(self.B, u[t])
            x[t+1] = self.semiring.complement(proj[t])*x[t] + proj[t]*update
            if debug:
                breakpoint()
        x = torch.stack(x)
        y = torch.einsum("yx,tx->ty", (self.C * self.pi), x[:-1].float())
        return SSMOutput(u, proj, x, y)

# Classes for trainable phonotactics models

class PhonotacticsModel(torch.nn.Module):
    def train(self,
              batches: Iterable[Iterable[Tuple[int, Sequence[int]]]], # iterable of batches of sequences of ints
              report_every: int=1000,
              debug: bool=False,
              reporting_window_size: int=100,
              checkpoint_prefix: Optional[str]=None,
              diagnostic_fns: Optional[Dict[str, Callable[[torch.nn.Module], Any]]]=None,
              hyperparams_to_report: Optional[Dict]=None,
              **kwds):
        opt = torch.optim.AdamW(params=self.parameters(), **kwds)
        reporting_window = deque(maxlen=reporting_window_size)
        if diagnostic_fns is None:
            diagnostic_fns = {}
        diagnostics = []
        writer = csv.DictWriter(
            sys.stdout,
            "step epoch mean_loss".split() + list(diagnostic_fns) + (
                list(hyperparams_to_report) if hyperparams_to_report else []
            )
        )
        writer.writeheader()
        for i, (epoch, batch) in enumerate(batches):
            opt.zero_grad()
            loss = -self.log_likelihood(batch, debug=debug).mean()
            loss.backward()
            opt.step()
            reporting_window.append(loss.detach())
            if i % report_every == 0:
                mean_loss = torch.stack(list(reporting_window)).mean().item()
                diagnostic = {
                    'step': i,
                    'epoch': epoch,
                    'mean_loss': mean_loss,
                }
                diagnostic |= {label : fn(self) for label, fn in diagnostic_fns.items()}
                if hyperparams_to_report:
                    diagnostic |= hyperparams_to_report
                writer.writerow(diagnostic)
                diagnostics.append(diagnostic)
                if checkpoint_prefix is not None:
                    filename = checkpoint_prefix + "_%d.pt" % i
                    with open(filename, 'wb') as outfile:
                        torch.save(self, outfile)
        return diagnostics


class FSAPhonotacticsModel(PhonotacticsModel):
    """ FSA Phonotactics model parameterized by log-space weights. """
    
    def __init__(self, A, init=None, final=None, phi=None, semiring=RealSemiring):
        super().__init__()
        self.A = A # unnormalized weights
        Q, S, Q2 = self.A.shape
        assert Q == Q2

        self.init = init # might be None
        self.final = final # might be None

        if phi is None:
            self.phi = torch.eye(S, device=DEVICE)
        else:
            self.phi = phi

        self.semiring = semiring            

    def A_positive(self):
        A_positive = self.semiring.from_log(self.A) # Q1 S Q2
        return A_positive

    def final_positive(self):
        return self.semiring.from_log(self.final)

    def forward(self, xss, debug=False):
        return self.log_likelihood(xss, debug=debug)

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

def soft_ceiling(x, k, beta=1):
    return k - torch.nn.functional.softplus(k - x, beta=beta)

def spectral_radius(A):
    return torch.linalg.eigvals(A).abs().max()

class GloballyNormalized:
    
    def log_likelihood(self, xs: Iterable[Sequence[int]], debug: Optional[bool]=False):
        wfsa = self.fsa()
        y = torch.stack([wfsa(x, debug=debug) for x in xs])
        logZ = self.fsa(semiring=RealSemiring).pathsum().log()
        return self.semiring.to_log(y) - logZ

    def fsa(self, semiring=None) -> WFSA:
        return WFSA(
            self.A_positive(),
            init=self.init,
            final=self.final_positive(),
            phi=self.phi,
            semiring=self.semiring if semiring is None else semiring,
        )

class AdjustedNormalized(GloballyNormalized):
    def fsa(self, semiring=None) -> WFSA:
        A_positive = self.semiring.to_exp(self.A_positive())
        if self.final is None:
            s = spectral_radius(A_positive.sum(-2))
            adjustment = soft_ceiling(s, 1) / s
            A_normalized = adjustment * A_positive
            final_normalized = None
        else:
            final_positive = self.semiring.to_exp(self.final_positive())
            s = spectral_radius(A_positive.sum(-2) + final_positive[:, None])
            adjustment = soft_ceiling(s, 1) / s
            A_normalized = adjustment * A_positive
            final_normalized = adjustment * final_positive
        return WFSA(
            self.semiring.from_exp(A_normalized),
            init=self.init,
            final=None if final_normalized is None else self.semiring.from_exp(final_normalized),
            phi=self.phi,
            semiring=self.semiring if semiring is None else semiring,
        )

class LocallyNormalized: # NOTE: PFSA models can have a halting probability, SSM models don't.
    def log_likelihood(self, xs: Iterable[Sequence[int]], debug: Optional[bool]=False):
        pfsa = self.fsa()
        y = torch.stack([pfsa(x, debug=debug) for x in xs])
        return self.semiring.to_log(y)

    def fsa(self, semiring=None) -> WFSA:
        A_positive = self.A_positive()
        if self.final is None:
            Z = self.semiring.sum(A_positive, dim=(-1, -2))
            A_normalized = self.semiring.div(A_positive, Z[:, None, None])
            final_normalized = None
        else:
            final_positive = self.final_positive()
            Z = self.semiring.add(self.semiring.sum(A_positive, dim=(-1, -2)), final_positive) # from each state, sum mass to halt or transition
            A_normalized = self.semiring.div(A_positive, Z[:, None, None] )
            final_normalized = self.semiring.div(final_positive, Z)
        return WFSA(
            A_normalized,
            init=self.init,
            final=final_normalized,
            semiring=self.semiring if semiring is None else semiring,
        )

class PFSAPhonotacticsModel(FSAPhonotacticsModel, LocallyNormalized):
    pass

class WFSAPhonotacticsModel(FSAPhonotacticsModel, AdjustedNormalized):
    pass

class pTSL(PFSAPhonotacticsModel):
    def __init__(self, E, pi, final=None, phi=None, semiring=RealSemiring):
        super(FSAPhonotacticsModel, self).__init__()
        self.E = E # shape QS
        self.pi = pi # shape S
        Q, S = self.E.shape
        assert S == self.pi.shape[-1]
        
        self.final = final
        self.init = torch.zeros(Q, device=DEVICE)
        self.init[0] = 1.0

        self.semiring = semiring

        # Both of these are stored in expspace
        self.T_on_tier = torch.cat([torch.zeros(1, S), torch.eye(S)]).T.to(DEVICE) # shape 1SQ, saving symbol as state
        self.T_not_on_tier = torch.eye(Q, device=DEVICE)[:, None, :] # shape Q1Q, preserving state

        if phi is None:
            self.phi = torch.eye(S, device=DEVICE)
        else:
            self.phi = phi
    
    @classmethod
    def initialize(cls,
                   S,
                   pi=None,
                   learn_final=False,
                   semiring=RealSemiring,
                   requires_grad=True,
                   init_T=DEFAULT_INIT_TEMPERATURE):
        if pi is None:
            pi = torch.nn.Parameter((1/init_T)*torch.randn(S), requires_grad=requires_grad)
        E = torch.nn.Parameter((1/init_T)*torch.randn(S+1, S), requires_grad=requires_grad)
        if learn_final:
            final = torch.nn.Parameter((1/init_T)*torch.randn(S+1), requires_grad=requires_grad)
            model = cls(E, pi, final=final, semiring=semiring)
        else:
            model = cls(E, pi, semiring=semiring)
        return model.to(DEVICE)

    def A_positive(self):
        # In real semiring:
        # A = proj * self.E.exp()[:, :, None] * self.T_on_tier + (1-proj) * self.T_not_on_tier
        proj = self.semiring.logistic(self.pi)[None, :, None]
        not_proj = self.semiring.logistic(-self.pi)[None, :, None]
        E = self.semiring.from_log(self.E)[:, :, None] 
        on_tier = self.semiring.mul(E, self.T_on_tier)
        one = self.semiring.mul(proj, on_tier)
        two = self.semiring.mul(not_proj, self.T_not_on_tier)
        A = self.semiring.add(one, two)
        return A

class SSMPhonotacticsModel(PhonotacticsModel):
    def __init__(self, A, B, C, init=None, pi=None):
        super().__init__()
        self.A = A
        self.B = B
        self.C = C
        self.init = init
        self.pi = pi

    @classmethod
    def initialize(
            cls,
            X,
            S,
            requires_grad=True,
            init_T_A=DEFAULT_INIT_TEMPERATURE,
            init_T_B=DEFAULT_INIT_TEMPERATURE,
            init_T_C=DEFAULT_INIT_TEMPERATURE):
        A = torch.nn.Parameter((1/init_T_A)*torch.randn(X, X), requires_grad=requires_grad)
        B = torch.nn.Parameter((1/init_T_B)*torch.randn(X, S), requires_grad=requires_grad)
        C = torch.nn.Parameter((1/init_T_C)*torch.randn(S, X), requires_grad=requires_grad)
        return cls(A, B, C).to(DEVICE)

    def log_likelihood(self, xs: Iterable[torch.LongTensor], debug: Optional[bool]=False):
        ssm = self.ssm()
        def gen():
            for x in xs:
                # for sequence x1 x2 x3, get the vector of logits of x_t | x_{<t}
                logits = ssm(x, debug=debug).y
                # normalize locally 
                yield logits.log_softmax(-1).gather(-1, x.unsqueeze(-1)).sum()
        return torch.stack(list(gen()))

    forward = log_likelihood

    def ssm(self) -> SSM:
        return SSM(self.A, self.B, self.C, init=self.init, pi=self.pi)

    def __add__(self, other):
        return CompoundSSMModel(self, other)

class DiagonalSSMPhonotacticsModel(SSMPhonotacticsModel):
    @classmethod
    def initialize(
            cls,
            X,
            S,
            requires_grad=True,
            A_diag=None,
            B=None,
            C=None,
            init_T_A=DEFAULT_INIT_TEMPERATURE,
            init_T_B=DEFAULT_INIT_TEMPERATURE,
            init_T_C=DEFAULT_INIT_TEMPERATURE):
        if A_diag is None:
            A_diag = torch.nn.Parameter((1/init_T_A)*torch.randn(X), requires_grad=requires_grad)
            
        if B is None:
            B = torch.nn.Parameter((1/init_T_B)*torch.randn(X, S), requires_grad=requires_grad)

        if C is None:
            C = torch.nn.Parameter((1/init_T_C)*torch.randn(S, X), requires_grad=requires_grad)
            
        return cls(A_diag, B, C).to(DEVICE)

    def ssm(self) -> SSM:
        return SSM(
            torch.diag(self.A),
            self.B,
            self.C,
            init=self.init,
            pi=self.pi
        )

class SquashedDiagonalSSMPhonotacticsModel(DiagonalSSMPhonotacticsModel):
    def ssm(self) -> SSM:
        return SSM(
            torch.diag(torch.sigmoid(self.A)),
            self.B,
            self.C,
            init=self.init,
            pi=self.pi
        )


class CompoundSSMModel(SSMPhonotacticsModel):
    def __init__(self, one, two):
        super(SSMPhonotacticsModel, self).__init__()
        self.one = one
        self.two = two

    def ssm(self) -> SSM:
        ssms = [m.ssm() for m in self.children()]
        return SSM(
            A=torch.block_diag(*[m.A for m in ssms]),
            B=torch.cat([m.B for m in ssms]),
            C=torch.cat([m.C for m in ssms], dim=-1),
            init=torch.cat([m.init for m in ssms]),
            pi=torch.cat([m.pi for m in ssms]),
        )

class Factor2(DiagonalSSMPhonotacticsModel):
    """ Phonotactics model whose probabilies are determined by factors of 2 segments """
    @classmethod
    def from_factors(cls, factors, projection=None, bias=False):
        S, X = factors.shape
        A, B, init = cls.init_matrices(X, bias=bias)
        return cls(A, B, factors, init=init, pi=projection)

    def init_matrices(self, d):
        raise NotImplementedError

    @classmethod
    def initialize(cls, S, projection=None, requires_grad: bool=True, init_T=DEFAULT_INIT_TEMPERATURE):
        factors = torch.nn.Parameter((1/init_T)*torch.randn(S, S+1), requires_grad=requires_grad)
        # S+1 for eos?
        return cls.from_factors(factors, projection=projection).to(DEVICE)


class SL2(Factor2):
    @classmethod
    def init_matrices(cls, d, bias=False):
        A_diag = torch.zeros(d+bias, device=DEVICE) 
        B = torch.eye(d+bias, device=DEVICE)[:, (1+bias):] # X x S
        init = torch.eye(d+bias, device=DEVICE)[0]
        if bias:
            A_diag[0] = 1
            init[0,1] = 1
        return A_diag, B, init

class SP2(Factor2):
    @classmethod
    def init_matrices(cls, d, bias=False):
        A_diag = torch.ones(d+bias, dtype=bool, device=DEVICE)
        B = torch.eye(d+bias, dtype=bool, device=DEVICE)[:, (1+bias):]
        init = torch.eye(d+bias, dtype=bool, device=DEVICE)[0]
        if bias:
            A_diag[0] = True
            init[0,1] = True
        return A_diag, B, init

class QuasiSP2(Factor2):
    @classmethod
    def init_matrices(cls, d, bias=False):
        A_diag = torch.ones(d+bias, device=DEVICE)
        B = torch.eye(d+bias, device=DEVICE)[:, (1+bias):]
        init = torch.eye(d+bias, device=DEVICE)[0]
        if bias:
            A_diag[0] = True
            init[0,1] = True
        return A_diag, B, init    

class SL_SP2(Factor2):
    @classmethod
    def init_matrices(cls, d):
        A_diag = torch.cat([
            torch.zeros(d, dtype=torch.bool),
            torch.ones(d, dtype=torch.bool),
        ]).to(DEVICE)
        B = torch.cat([
            torch.eye(d, dtype=torch.bool)[:, 1:],
            torch.eye(d, dtype=torch.bool)[:, 1:],
        ]).to(DEVICE)
        init = torch.cat([
            torch.eye(d, dtype=torch.bool)[0],
            torch.eye(d, dtype=torch.bool)[0],
        ]).to(DEVICE)
        return A_diag, B, init

class TierBased(Factor2):
    """ Assume projection is a function from input features to a single number for all state features,
    that is, we can say that a segment is or is not projected. """

    def ssm(self):
        return SSM(
            torch.diag(self.A),
            self.B,
            self.C,
            init=self.init,
            pi=self.pi.unsqueeze(-1).expand(*self.C.shape), # broadcast projection to all state features
        )

class SoftTierBased(Factor2):
    @classmethod
    def initialize(
            cls,
            S,
            pi=None, 
            requires_grad: bool=True,
            init_T=DEFAULT_INIT_TEMPERATURE,
            init_T_projection=DEFAULT_INIT_TEMPERATURE):
        factors = torch.nn.Parameter((1/init_T)*torch.randn(S, S+1), requires_grad=requires_grad)
        if pi is None:
            projection = torch.nn.Parameter((1/init_T_projection)*torch.randn(S), requires_grad=requires_grad)
        else:
            projection = pi
        return cls.from_factors(factors, projection).to(DEVICE)

    def ssm(self):
        return SSM(
            torch.diag(self.A),
            self.B,
            self.C,
            init=self.init,
            pi=self.pi.sigmoid().unsqueeze(-1).expand(*self.C.shape), # squash projection probabilities to (0,1)
        )

class ProbabilisticTierBased(SoftTierBased):
    def ssm(self):
        return SSM(
            torch.diag(self.A),
            self.B,
            self.C.exp(),
            init=self.init,
            pi=self.pi.sigmoid().unsqueeze(-1).expand(*self.C.shape),
        )

    def log_likelihood(self, xs: Iterable[torch.LongTensor], debug: Optional[bool]=False):
        ssm = self.ssm()
        def gen():
            for x in xs:
                weights = ssm(x).y + ssm.semiring.complement(ssm.pi[:,x][0])[:, None] # hack!!!
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
    dataset = [random_no_axb(n=n) for _ in range(num_samples)]
    # the first two of these comparisons will be exactly zero for SL
    # the third comparison will be exactly zero for SL and SP

    good = [ # no 23 on the tier {1, 2, 3}
         [0,2,0,1,0,3], # matched on SL factors
         [0,0,2,0,0,1,0,0,3], # matched on SL factors
         [0,2,0,1,0,2,0,1,0,3], # matched on (boolean) SP factors: 00,01,02,03,10,11,12,13,20,21,22,23
         [1,1,3,0,2,0],
         [3,3,2,1,3,0],
    ]

    bad = [
        [0,1,0,2,0,3],
        [0,0,1,0,0,2,0,0,3],
        [0,2,0,1,0,1,0,2,0,3], # same SP factors as the SP example above;
        [1,1,2,0,3,0],
        [3,3,2,0,3,0],
    ]
    #good = [random_no_axb(n=n+1) for _ in range(num_test)]
    #bad = [random_no_axb(n=n+1, neg=True) for _ in range(num_test)]

    if isinstance(model_type, TierBased): # fixed tier -- the right projection function is [0,1,1,1].
        model = model_type.initialize(4, torch.Tensor([0,1,1,1])) 
    elif force_pi is not None:
        model = model_type.initialize(4, pi=force_pi)
    else:
        model = model_type.initialize(4)

    diagnostics = {'good_bad_diff': lambda m: evaluate_model_paired(m, good, bad)['diff'].sum()}
    data = minibatches(dataset, batch_size, num_epochs=num_epochs)
    model.train(data, diagnostic_fns=diagnostics, **kwds)
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
