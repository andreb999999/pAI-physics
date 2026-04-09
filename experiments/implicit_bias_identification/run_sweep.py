#!/usr/bin/env python3
"""
Phase 1: Systematic Implicit Bias Regularizer Identification for Muon

Generates random matrix sensing instances, runs Muon/AdamW/GD to convergence,
then compares converged solutions against CVXPY solutions for candidate regularizers.

Usage:
    python run_sweep.py --instance-id 0 --total-instances 500
    (designed for SLURM array jobs)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---- Reproducibility ----
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---- Matrix Sensing Problem ----
class MatrixSensingProblem:
    """
    Generate: A_i ∈ R^{m×d}, W* ∈ R^{m×d}, b_i = tr(A_i^T W*)
    Objective: minimize ‖A(W) - b‖² where A(W)_i = tr(A_i^T W)
    """
    def __init__(self, m, d, p, condition_number=1.0, seed=42):
        set_seed(seed)
        self.m = m
        self.d = d
        self.p = p  # number of measurements

        # Generate ground truth with controlled condition number
        U, _, Vt = np.linalg.svd(np.random.randn(m, d), full_matrices=False)
        r = min(m, d)
        # Singular values from 1 to 1/condition_number
        sigmas = np.linspace(1.0, 1.0 / max(condition_number, 1.0), r)
        self.W_star = U[:, :r] @ np.diag(sigmas) @ Vt[:r, :]

        # Measurement matrices
        self.A = np.random.randn(p, m, d) / np.sqrt(m * d)

        # Observations
        self.b = np.array([np.trace(self.A[i].T @ self.W_star) for i in range(p)])

    def loss(self, W):
        """Compute ‖A(W) - b‖²"""
        predictions = np.array([np.trace(self.A[i].T @ W) for i in range(self.p)])
        return 0.5 * np.sum((predictions - self.b) ** 2)

    def gradient(self, W):
        """Compute gradient of ‖A(W) - b‖²"""
        predictions = np.array([np.trace(self.A[i].T @ W) for i in range(self.p)])
        residuals = predictions - self.b
        grad = sum(residuals[i] * self.A[i] for i in range(self.p))
        return grad

    def to_torch(self, device='cuda'):
        return {
            'A': torch.tensor(self.A, dtype=torch.float32, device=device),
            'b': torch.tensor(self.b, dtype=torch.float32, device=device),
            'W_star': torch.tensor(self.W_star, dtype=torch.float32, device=device),
        }


# ---- Muon Optimizer (Polar Map) ----
class Muon(torch.optim.Optimizer):
    """Muon: steepest descent under the polar map geometry."""
    def __init__(self, params, lr=0.01, momentum=0.0):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                # Polar map: project gradient onto orthogonal component
                U, S, Vt = torch.linalg.svd(g, full_matrices=False)
                # Muon update: use the polar factor U @ Vt
                polar = U @ Vt

                if momentum > 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(polar)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(polar)
                    p.add_(buf, alpha=-lr)
                else:
                    p.add_(polar, alpha=-lr)
        return loss


# ---- Run a single optimization ----
def run_optimizer(problem_torch, optimizer_name, lr, momentum=0.0, max_steps=100000, tol=1e-12, device='cuda'):
    A = problem_torch['A']
    b = problem_torch['b']
    p, m, d = A.shape

    W = nn.Parameter(torch.zeros(m, d, device=device) + 0.001 * torch.randn(m, d, device=device))

    if optimizer_name == 'muon':
        opt = Muon([W], lr=lr, momentum=momentum)
    elif optimizer_name == 'adamw':
        opt = torch.optim.AdamW([W], lr=lr, weight_decay=0)
    elif optimizer_name == 'gd':
        opt = torch.optim.SGD([W], lr=lr, momentum=0)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    best_loss = float('inf')
    patience_counter = 0

    for step in range(max_steps):
        opt.zero_grad()
        # Compute loss
        pred = torch.einsum('pmd,md->p', A, W)
        loss = 0.5 * torch.sum((pred - b) ** 2)

        if loss.item() < tol:
            break

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 5000:
                break  # Converged (loss not decreasing)

        loss.backward()
        opt.step()

    return W.detach().cpu().numpy(), best_loss, step


# ---- Compute regularizer metrics ----
def compute_metrics(W):
    """Compute all candidate regularizer values for a matrix W."""
    U, sigmas, Vt = np.linalg.svd(W, full_matrices=False)
    sigmas = sigmas[sigmas > 1e-10]  # Filter numerical zeros

    metrics = {}
    metrics['nuclear_norm'] = np.sum(sigmas)
    metrics['operator_norm'] = np.max(sigmas)
    metrics['frobenius_norm'] = np.sqrt(np.sum(sigmas ** 2))

    # Spectral entropy
    p_sigma = sigmas / np.sum(sigmas)
    metrics['spectral_entropy'] = -np.sum(p_sigma * np.log(p_sigma + 1e-15))
    metrics['neg_spectral_entropy'] = -metrics['spectral_entropy']

    # Log-determinant (of W^T W)
    if len(sigmas) > 0 and np.min(sigmas) > 1e-10:
        metrics['neg_log_det'] = -np.sum(np.log(sigmas ** 2))
    else:
        metrics['neg_log_det'] = float('inf')

    # Condition number
    if len(sigmas) > 0 and sigmas[-1] > 1e-10:
        metrics['condition_number'] = sigmas[0] / sigmas[-1]
    else:
        metrics['condition_number'] = float('inf')

    # Spectral Wasserstein to uniform
    r = len(sigmas)
    if r > 0:
        uniform = np.full(r, np.mean(sigmas))
        sorted_s = np.sort(sigmas)
        sorted_u = np.sort(uniform)
        metrics['spectral_wasserstein'] = np.sqrt(np.mean((sorted_s - sorted_u) ** 2))
    else:
        metrics['spectral_wasserstein'] = float('inf')

    # Raw singular values
    metrics['singular_values'] = sigmas.tolist()

    return metrics


# ---- CVXPY solutions ----
def solve_cvxpy(problem, regularizer='nuclear_norm'):
    """Solve min R(W) s.t. A(W) = b using CVXPY."""
    try:
        import cvxpy as cp
    except ImportError:
        return None, None

    m, d = problem.m, problem.d
    W = cp.Variable((m, d))

    # Constraints: A(W) = b
    constraints = []
    for i in range(problem.p):
        constraints.append(cp.trace(problem.A[i].T @ W) == problem.b[i])

    if regularizer == 'nuclear_norm':
        objective = cp.Minimize(cp.normNuc(W))
    elif regularizer == 'operator_norm':
        objective = cp.Minimize(cp.norm(W, 2))
    elif regularizer == 'frobenius_norm':
        objective = cp.Minimize(cp.norm(W, 'fro'))
    elif regularizer == 'neg_log_det':
        # min -log det(W^T W) is equivalent to max log det
        # Use log_det approximation: min -log det(W^T W + εI)
        eps = 1e-6
        objective = cp.Minimize(-cp.log_det(W.T @ W + eps * np.eye(d)))
    else:
        return None, None

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, max_iters=50000, verbose=False)
        if prob.status in ('optimal', 'optimal_inaccurate'):
            return W.value, prob.value
    except Exception as e:
        print(f"CVXPY failed for {regularizer}: {e}")

    return None, None


# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance-id', type=int, required=True)
    parser.add_argument('--total-instances', type=int, default=500)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    idx = args.instance_id
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Instance parameters — systematic sweep
    # UNDER-DETERMINED regime: p < m*d so multiple interpolating solutions exist
    # This is where implicit bias matters — the optimizer's geometry determines WHICH solution
    dims = [(10, 10), (10, 20), (20, 20), (20, 40), (30, 30), (30, 60)]
    measurement_fractions = [0.2, 0.3, 0.5, 0.7]  # p = fraction * m * d
    condition_numbers = [1.0, 2.0, 5.0, 10.0, 50.0]

    configs = []
    for m, d in dims:
        for frac in measurement_fractions:
            for kappa in condition_numbers:
                p = max(int(frac * m * d), min(m, d) + 1)  # At least rank+1 measurements
                configs.append((m, d, p, kappa))

    if idx >= len(configs):
        print(f"Instance {idx} out of range ({len(configs)} configs). Skipping.")
        return

    m, d, p, kappa = configs[idx]
    seed = 42 + idx

    print(f"Instance {idx}: m={m}, d={d}, p={p}, kappa={kappa}, seed={seed}")

    # Generate problem
    problem = MatrixSensingProblem(m, d, p, condition_number=kappa, seed=seed)
    problem_torch = problem.to_torch(device=args.device)

    results = {
        'instance_id': idx,
        'config': {'m': m, 'd': d, 'p': p, 'kappa': kappa, 'seed': seed},
        'optimizers': {},
        'cvxpy': {},
    }

    # Run each optimizer with lr sweep
    for opt_name in ['muon', 'adamw', 'gd']:
        lr_candidates = {
            'muon': [0.1, 0.05, 0.01, 0.005, 0.001],
            'adamw': [0.01, 0.005, 0.001, 0.0005, 0.0001],
            'gd': [0.1, 0.05, 0.01, 0.005, 0.001],
        }

        best_result = None
        best_loss = float('inf')

        for lr in lr_candidates[opt_name]:
            try:
                W_opt, loss, steps = run_optimizer(
                    problem_torch, opt_name, lr=lr,
                    momentum=0.95 if opt_name == 'muon' else 0.0,
                    device=args.device,
                )
                if loss < best_loss:
                    best_loss = loss
                    best_result = {
                        'lr': lr,
                        'loss': float(loss),
                        'steps': steps,
                        'metrics': compute_metrics(W_opt),
                        'W': W_opt.tolist(),
                    }
            except Exception as e:
                print(f"  {opt_name} lr={lr} failed: {e}")

        if best_result:
            # Don't store full W in JSON (too large), just metrics
            W_arr = np.array(best_result.pop('W'))
            results['optimizers'][opt_name] = best_result
            # Save W separately as numpy
            np.save(out_dir / f'W_{opt_name}_{idx:04d}.npy', W_arr)

    # Solve CVXPY for candidate regularizers
    for reg_name in ['nuclear_norm', 'operator_norm', 'frobenius_norm', 'neg_log_det']:
        print(f"  CVXPY: {reg_name}...")
        W_cvx, val = solve_cvxpy(problem, reg_name)
        if W_cvx is not None:
            results['cvxpy'][reg_name] = {
                'objective': float(val),
                'metrics': compute_metrics(W_cvx),
            }
            np.save(out_dir / f'W_cvxpy_{reg_name}_{idx:04d}.npy', W_cvx)

    # Compute distances between Muon solution and CVXPY solutions
    if 'muon' in results['optimizers']:
        W_muon = np.load(out_dir / f'W_muon_{idx:04d}.npy')
        muon_norm = np.linalg.norm(W_muon)
        results['distances'] = {}
        for reg_name in results['cvxpy']:
            W_cvx = np.load(out_dir / f'W_cvxpy_{reg_name}_{idx:04d}.npy')
            dist = np.linalg.norm(W_muon - W_cvx) / max(muon_norm, 1e-10)
            results['distances'][reg_name] = float(dist)

    # Save results (convert numpy types to native Python for JSON)
    def _sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    out_file = out_dir / f'result_{idx:04d}.json'
    with open(out_file, 'w') as f:
        json.dump(_sanitize(results), f, indent=2)

    print(f"Done. Results saved to {out_file}")
    print(f"  Distances to CVXPY: {results.get('distances', {})}")


if __name__ == '__main__':
    main()
