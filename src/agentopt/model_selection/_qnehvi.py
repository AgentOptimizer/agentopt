"""Optional discrete qNEHVI acquisition for live Pareto selection."""

import numpy as np


def require_botorch():
    try:
        import torch  # noqa: F401
        from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement  # noqa: F401
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP  # noqa: F401
    except ImportError as error:
        raise ImportError('Cost-aware qNEHVI requires `pip install "agentopt-py[bayesian]"`') from error


def select_qnehvi_index(
    train_features, train_objectives, candidate_features, *, categorical_dims,
    baseline_features, candidate_costs, mc_samples=64, seed=0, candidate_chunk_size=64,
):
    """Fit independent categorical GPs and score q=1 candidates per dollar.

    Candidate chunks share one acquisition and QMC sampler. CPU RNG state is
    restored afterwards; the library does not change global Torch thread limits.
    """
    require_botorch()
    import torch
    from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

    train_x = np.asarray(train_features, dtype=float)
    train_y = np.asarray(train_objectives, dtype=float)
    candidate_x = np.asarray(candidate_features, dtype=float)
    baseline_x = np.asarray(baseline_features, dtype=float)
    costs = np.asarray(candidate_costs, dtype=float)
    if train_x.ndim != 2 or not all(train_x.shape) or train_y.shape != (len(train_x), 2):
        raise ValueError("Training features must be nonempty N-by-D and objectives N-by-2")
    if candidate_x.ndim != 2 or not len(candidate_x) or candidate_x.shape[1] != train_x.shape[1]:
        raise ValueError("Candidate features must match training feature columns")
    if baseline_x.ndim != 2 or baseline_x.shape[1] != train_x.shape[1]:
        raise ValueError("Baseline features must match training feature columns")
    if not all(np.all(np.isfinite(x)) for x in (train_x, train_y, candidate_x, baseline_x)):
        raise ValueError("GP features and objectives must be finite")
    if costs.shape != (len(candidate_x),) or not np.all(np.isfinite(costs) & (costs > 0)):
        raise ValueError("Candidate costs must be finite and positive")

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed) % (2**31))
        x = torch.as_tensor(train_x, dtype=torch.float64)
        y = torch.as_tensor(train_y, dtype=torch.float64)
        model = ModelListGP(*[
            MixedSingleTaskGP(
                train_X=x, train_Y=y[:, i:i + 1], cat_dims=categorical_dims,
                outcome_transform=Standardize(m=1),
            )
            for i in range(2)
        ])
        fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]), seed=int(seed) % (2**31))
        acquisition = qNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=[0.0, 0.0],
            X_baseline=torch.as_tensor(baseline_x, dtype=torch.float64),
            sampler=sampler, prune_baseline=bool(len(baseline_x)),
        )
        candidates = torch.as_tensor(candidate_x, dtype=torch.float64).unsqueeze(1)
        with torch.no_grad():
            values = torch.cat([
                acquisition(chunk).reshape(-1)
                for chunk in candidates.split(candidate_chunk_size, dim=0)
            ])
        scores = values.cpu().numpy()
    if scores.shape != costs.shape or not np.all(np.isfinite(scores)):
        raise RuntimeError("qNEHVI returned invalid acquisition scores")
    return int(np.argmax(scores / costs))
