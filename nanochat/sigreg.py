"""
SIGReg (Sketched Isotropic Gaussian Regularization), adapted for language modelling.

Background
----------
SIGReg is the regularizer introduced in LeJEPA (Balestriero & LeCun) and reused as one of
the two loss terms in LeWorldModel. In a JEPA the only other term is a prediction loss,
which is trivially minimized by a constant embedding ("collapse"). SIGReg removes that
degenerate solution by pushing the *distribution* of embeddings towards an isotropic
Gaussian N(0, I_d), which by construction has full-rank covariance.

Testing d-dimensional Gaussianity directly is expensive and high variance. SIGReg gets
around it with two ideas:

1. Cramer-Wold: a distribution on R^d is N(0, I_d) iff *every* 1-D projection <x, a> is
   N(0, 1). So we sample M random unit directions ("slices") and only ever run a
   univariate test. Directions are resampled every step, so over training we cover the
   sphere while paying O(N*M) per step.

2. Epps-Pulley: a univariate normality test that compares the *empirical characteristic
   function* of the projections against the CF of N(0,1), which is known in closed form,
   phi(t) = exp(-t^2/2):

       T_n = n * integral | ECF_n(t) - exp(-t^2/2) |^2 w(t) dt,   w(t) = exp(-t^2/2)

   with ECF_n(t) = (1/n) sum_j exp(i t z_j). Writing the modulus out in real terms:

       | ECF_n(t) - phi(t) |^2 = ( mean_j cos(t z_j) - phi(t) )^2 + ( mean_j sin(t z_j) )^2

   The integral is done by trapezoid over K knots on [0, t_max]; because both the ECF
   error and w are even in t we integrate on the half-line and double, which is the same
   quadrature for half the knots. Unlike moment- or sorting-based tests (skew/kurtosis,
   KS, Shapiro-Wilk) this statistic and its gradients are bounded, which is what makes it
   safe to backprop through for millions of steps.

Adapting it to a language model
-------------------------------
The reference implementation regularizes one embedding vector per image view. A causal LM
produces one hidden state per *token*, so a batch is (B, T, C) and the natural sample set
is the B*T token vectors. Two things change:

- Independence. Epps-Pulley's n-scaling assumes i.i.d. samples. Tokens inside a sequence
  are strongly dependent, so the effective sample size is well below B*T and the raw
  statistic is inflated. `token_subsample` draws a random subset of token positions each
  step, which cuts cost and decorrelates the sample.

- What to make Gaussian. `center` chooses the quantity the test is applied to. 'none' is
  the LeJEPA/LeWM default (the latent marginal). 'sequence' subtracts each sequence's own
  mean first, so only the within-sequence *residual* is shaped and the sequence-level mean
  is left free. That mirrors the "temporally centered SIGReg" variant found to help on
  multi-task LeWorldModel, and it matters much more for an LM than for images: a big part
  of the marginal spread of LM hidden states is document/topic identity we do not want to
  whiten away.

References: LeJEPA arXiv:2511.08544; the quadrature-folding trick and the reference
~10-line forward come from the LeJEPA MINIMAL.md walkthrough.
"""

import torch
import torch.nn as nn


def epps_pulley_statistic(z, t, phi, weights):
    """
    Epps-Pulley statistic for each of M slices.

    z:       (N, M) projections of N samples onto M unit directions
    t:       (K,)   quadrature knots
    phi:     (K,)   CF of N(0,1) at the knots, exp(-t^2/2)
    weights: (K,)   quadrature weights, already multiplied by the window w(t)
    returns: (M,)   statistic per slice, NOT yet scaled by N
    """
    zt = z.unsqueeze(-1) * t                                     # (N, M, K)
    err = (zt.cos().mean(-3) - phi).square() + zt.sin().mean(-3).square()  # (M, K)
    return err @ weights                                         # (M,)


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization for (B, T, C) hidden states.

    Args:
        num_slices:      M, random 1-D directions per step. More slices = lower variance
                         estimate of the same objective, not a different objective.
        knots:           K, quadrature points on [0, t_max]. 17 is the reference value.
        t_max:           upper limit of the CF integral. exp(-t^2/2) has already decayed to
                         ~0.011 by t=3, so there is nothing left to integrate past it.
        token_subsample: max token vectors used per step (None = use all B*T).
        center:          'none' | 'sequence' | 'batch', see module docstring.
        scale_by_n:      multiply by N as in the classical statistic. Faithful to the
                         reference, but it makes the loss magnitude depend on N, so lambda
                         has to be retuned if you change the token count.
    """

    def __init__(self, num_slices=256, knots=17, t_max=3.0, token_subsample=None,
                 center='none', scale_by_n=True):
        super().__init__()
        assert center in ('none', 'sequence', 'batch')
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2 * dt, dtype=torch.float32)  # trapezoid, doubled by symmetry
        w[[0, -1]] = dt                                        # endpoints count once
        window = torch.exp(-t.square() / 2.0)                  # w(t), and also the CF of N(0,1)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", w * window)
        self.num_slices = num_slices
        self.token_subsample = token_subsample
        self.center = center
        self.scale_by_n = scale_by_n

    def forward(self, x, generator=None):
        """
        x: (B, T, C) hidden states, or (N, C) if already flattened.
        generator: optional torch.Generator for the slice/token sampling. Pass a dedicated
                   one so that turning SIGReg on does not shift the global RNG stream and
                   silently change the baseline's data order or init.
        """
        if x.dim() == 3:
            if self.center == 'sequence':
                x = x - x.mean(dim=1, keepdim=True)
            x = x.reshape(-1, x.size(-1))
        assert x.dim() == 2, f"expected (B,T,C) or (N,C), got {tuple(x.shape)}"
        if self.center == 'batch':
            x = x - x.mean(dim=0, keepdim=True)

        x = x.float()  # cos/sin means are cancellation-heavy, keep them out of bf16
        N, C = x.shape

        if self.token_subsample is not None and self.token_subsample < N:
            idx = torch.randperm(N, device=x.device, generator=generator)[:self.token_subsample]
            x = x[idx]
            N = x.size(0)

        # Fresh random slices every step: an unbiased Monte Carlo estimate of the
        # Cramer-Wold objective over all directions.
        a = torch.randn(C, self.num_slices, device=x.device, dtype=x.dtype, generator=generator)
        a = a / a.norm(p=2, dim=0, keepdim=True)

        stat = epps_pulley_statistic(x @ a, self.t, self.phi, self.weights)
        if self.scale_by_n:
            stat = stat * N
        return stat.mean()
