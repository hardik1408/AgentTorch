"""Differentiable Discrete Distributions for Dynamics and Interventions"""

import numpy as np
import torch
import torch.nn as nn


class StraightThroughBernoulli(torch.autograd.Function):
    generate_vmap_rule = True

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, p):
        result = torch.bernoulli(p)
        ctx.save_for_backward(result, p)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, p = ctx.saved_tensors
        ws = torch.ones(result.shape)
        return grad_output * ws


class Bernoulli(torch.autograd.Function):
    generate_vmap_rule = True

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, p):
        result = torch.bernoulli(p)
        ctx.save_for_backward(result, p)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, p = ctx.saved_tensors
        w_minus = (1.0 / p) / 2  # jump down, averaged for eps > 0 and eps < 0
        w_plus = (1.0 / (1.0 - p)) / 2  # jump up, averaged for eps > 0 and eps < 0

        ws = torch.where(
            result == 1, w_minus, w_plus
        )  # stochastic triple: total grad -> + smoothing rule)
        return grad_output * ws


class Binomial(torch.autograd.Function):
    generate_vmap_rule = True

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, n, p):
        result = torch.distributions.binomial.Binomial(n, p).sample()
        ctx.save_for_backward(result, p, n)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, p, n = ctx.saved_tensors
        w_minus = result / p  # derivative contributions of unit jump down
        w_plus = (n - result) / (1.0 - p)  # derivative contributions of unit jump up

        wminus_cont = torch.where(result > 0, w_minus, 0)
        wplus_cont = torch.where(result < n, w_plus, 0)

        ws = (wminus_cont + wplus_cont) / 2

        return None, grad_output * ws
    

class Geometric(torch.autograd.Function):
    generate_vmap_rule = True

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, p):
        p = torch.clamp(p, min=1e-5, max=1 - 1e-5)  # avoid numerical issues

        u = torch.rand(p.shape) 
        result = torch.ceil(torch.log(1 - u) / torch.log(1 - p))  # Inverse CDF method

        ctx.save_for_backward(result, p)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, p = ctx.saved_tensors

        w_plus = (1.0 / p)  # weight for jumping up
        w_minus = ((result - 1.0) / (1.0 - p))  # weight for jumping down

        w_plus_cont = torch.where(result > 0, w_plus, torch.zeros_like(w_plus))
        w_minus_cont = torch.where(result > 1, w_minus, torch.zeros_like(w_minus))

        ws = (w_plus_cont + w_minus_cont) / 2.0  # average weights for unbiased gradientn
        return grad_output * ws
    

class StochasticCategorical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs):
        """
        Forward pass: Sample a category (index) from a categorical distribution.
        
        Args:
            probs (Tensor): 1D tensor of probabilities (summing to 1) with requires_grad=True.
        
        Returns:
            Tensor: A scalar float tensor equal to the sampled index.
        """
        # Compute the cumulative distribution
        cum_probs = torch.cumsum(probs, dim=0)
        # Sample a uniform number in [0,1)
        u = torch.rand(1, device=probs.device)
        # Find the first index where u <= cumulative probability
        idx = torch.searchsorted(cum_probs, u)
        idx_int = int(idx.item())
        # Save for backward: the probabilities and the sampled index.
        ctx.save_for_backward(probs, torch.tensor(idx_int, device=probs.device))
        # The “primal” outcome is just the sampled index; convert to float so gradients flow.
        return torch.tensor(idx_int, dtype=torch.float32, device=probs.device)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Return an unbiased derivative estimate via the discrete jump.
        
        For the sampled category i, we set:
          - If i > 0: a “downward jump” is applied, using the cumulative probability below index i,
                F_lower = sum(probs[0:i]).
              Then we define w = F_lower / probs[i] and Y = i – 1.
              Thus the derivative estimate is w · (Y – i) = -w.
          - If i == 0, no jump is available so the derivative is 0.
        
        Args:
            grad_output (Tensor): Upstream gradient.
        
        Returns:
            Tensor: Gradient with respect to `probs` (a vector with zeros everywhere except possibly at one entry).
        """
        probs, idx_tensor = ctx.saved_tensors
        i = int(idx_tensor.item())
        grad_probs = torch.zeros_like(probs)
        n = probs.shape[0]
        # For i==0, no upward (or downward) jump is available.
        if i == 0:
            return grad_probs
        # Compute the cumulative probability below the sampled index:
        F_lower = torch.sum(probs[:i])
        # Set jump weight: w = F_lower / probs[i]
        w = F_lower / probs[i]
        # Alternative outcome: Y = i - 1; therefore, the discrete derivative equals
        # w * (Y - i) = w * ((i - 1) - i) = -w.
        # Propagate the (scalar) gradient only to the i-th element.
        grad_probs[i] = grad_output * (-w)
        return grad_probs












