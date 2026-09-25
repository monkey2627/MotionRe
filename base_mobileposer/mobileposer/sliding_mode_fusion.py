"""Sliding-mode-observer (SMO) fusion of IMU-only translation with the
synthetic waist-camera SLAM position -- a deterministic, control-theory
alternative to CameraFusionGRU's learned trust gate (see camera_fusion.py).

Novelty framing: sliding-mode observers are a mature tool in attitude
estimation (e.g. "Sliding mode observer to estimate both the attitude and
the gyro-bias by using low-cost sensors") and GPS-denied camera/inertial
navigation, prized for finite-time, disturbance-bounded convergence
guarantees that don't rely on a Gaussian-noise assumption the way a Kalman
gain or a learned network's implicit prior does. As far as searched
(2026-09-25), this has not been applied to sparse-IMU human pose/translation
reconstruction (DIP/TransPose/PIP/MobilePoser lineage) -- CameraFusionGRU
solves the identical fusion problem with a learned GRU gate; this module is
a from-a-different-field alternative to that same problem, not a
modification of it.

Discrete-time super-twisting algorithm (Levant), the standard modern
(chattering-reduced, finite-time-convergent) second-order sliding-mode
observer design, applied per-axis to the position residual between the two
signals:

    e[t]     = slam_pos[t] - p_hat[t]                      (sliding variable)
    p_hat[t] = p_hat[t-1] + imu_disp[t] + z[t-1] + lam*sqrt(|e[t-1]|)*sign(e[t-1])
    z[t]     = z[t-1] + alpha*sign(e[t-1])*dt

`imu_disp` (the frozen network's own frame-to-frame displacement) plays the
role of the known/modeled propagation term; the sqrt+sign proportional term
and the integral `z` (the standard super-twisting structure) pull the
estimate toward the SLAM measurement with a robust, non-Kalman correction
law. Only two scalar gains (lam, alpha) -- tuned by grid search on the TRAIN
fold, no gradient descent, no learned weights.
"""
import torch


def _as_gain_tensor(g, B, dtype, device):
    """Accept a scalar or a length-B tensor/list of gains; return [B,1] for broadcasting."""
    if torch.is_tensor(g):
        t = g.to(device=device, dtype=dtype)
    else:
        t = torch.full((B,), float(g), dtype=dtype, device=device)
    return t.view(B, 1)


def sliding_mode_fuse(imu_pos: torch.Tensor, slam_pos: torch.Tensor,
                       lam, alpha, fps: float = 30.0) -> torch.Tensor:
    """imu_pos, slam_pos: [T,3] or [B,T,3]. lam/alpha: scalar, or length-B
    tensor/list to evaluate B different gain settings in one vectorized pass
    (e.g. imu_pos/slam_pos broadcast-expanded to [B,T,3] for a grid search --
    see grid_search_gains). Returns fused position, same leading shape."""
    single = imu_pos.dim() == 2
    if single:
        imu_pos = imu_pos.unsqueeze(0)
        slam_pos = slam_pos.unsqueeze(0)
    if imu_pos.shape != slam_pos.shape or imu_pos.shape[-1] != 3:
        raise ValueError('Expected imu_pos/slam_pos of shape [B,T,3]')

    B, T, _ = imu_pos.shape
    dt = 1.0 / fps
    lam_t = _as_gain_tensor(lam, B, imu_pos.dtype, imu_pos.device)
    alpha_t = _as_gain_tensor(alpha, B, imu_pos.dtype, imu_pos.device)
    imu_disp = torch.zeros_like(imu_pos)
    imu_disp[:, 1:] = imu_pos[:, 1:] - imu_pos[:, :-1]

    p_hat = torch.zeros_like(imu_pos)
    p_hat[:, 0] = imu_pos[:, 0]
    z = torch.zeros(B, 3, dtype=imu_pos.dtype, device=imu_pos.device)
    e_prev = slam_pos[:, 0] - p_hat[:, 0]

    for t in range(1, T):
        correction = lam_t * torch.sqrt(e_prev.abs() + 1e-9) * torch.sign(e_prev)
        p_hat[:, t] = p_hat[:, t - 1] + imu_disp[:, t] + z + correction
        z = z + alpha_t * torch.sign(e_prev) * dt
        e_prev = slam_pos[:, t] - p_hat[:, t]

    return p_hat.squeeze(0) if single else p_hat


def sliding_mode_fuse_bias(imu_pos: torch.Tensor, slam_pos: torch.Tensor,
                            lam, alpha, fps: float = 30.0) -> torch.Tensor:
    """Bias-observer form of the same super-twisting SMO (the structure actually
    used in the cited literature, e.g. attitude+gyro-bias sliding-mode observers):
    estimate the slowly-varying drift BIAS between imu_pos and slam_pos directly
    from the residual, then subtract it from imu_pos each frame -- instead of
    recursively injecting the correction into an accumulating position state
    (sliding_mode_fuse above), which is prone to integrator windup/overshoot
    compounding over thousands of frames.

        e[t]     = (imu_pos[t] - b_hat[t-1]) - slam_pos[t]
        b_hat[t] = b_hat[t-1] + alpha*sign(e[t])*dt + lam*sqrt(|e[t]|)*sign(e[t])
        p_hat[t] = imu_pos[t] - b_hat[t]

    p_hat is anchored to the current-frame IMU reading every step, so estimation
    errors in b_hat cannot compound across frames the way they could in the
    recursive-position version. lam/alpha: scalar or length-B, see sliding_mode_fuse.
    """
    single = imu_pos.dim() == 2
    if single:
        imu_pos = imu_pos.unsqueeze(0)
        slam_pos = slam_pos.unsqueeze(0)
    if imu_pos.shape != slam_pos.shape or imu_pos.shape[-1] != 3:
        raise ValueError('Expected imu_pos/slam_pos of shape [B,T,3]')

    B, T, _ = imu_pos.shape
    dt = 1.0 / fps
    lam_t = _as_gain_tensor(lam, B, imu_pos.dtype, imu_pos.device)
    alpha_t = _as_gain_tensor(alpha, B, imu_pos.dtype, imu_pos.device)
    b_hat = torch.zeros(B, 3, dtype=imu_pos.dtype, device=imu_pos.device)
    p_hat = torch.zeros_like(imu_pos)
    p_hat[:, 0] = imu_pos[:, 0]

    for t in range(1, T):
        e = (imu_pos[:, t] - b_hat) - slam_pos[:, t]
        b_hat = b_hat + alpha_t * torch.sign(e) * dt + lam_t * torch.sqrt(e.abs() + 1e-9) * torch.sign(e)
        p_hat[:, t] = imu_pos[:, t] - b_hat

    return p_hat.squeeze(0) if single else p_hat


def grid_search_gains(train_seqs, slam_rms_m, lam_grid, alpha_grid, fps=30.0, seed_base=0,
                       fuse_fn=None, device=None):
    """train_seqs: list of (subset, source, tran_pred, tran_gt). Returns best (lam, alpha, mean_err).

    Vectorized across the whole (lam, alpha) grid: for each sequence, all
    G=len(lam_grid)*len(alpha_grid) gain combinations are evaluated in a single
    batched forward pass (batch dim = G) instead of G separate per-sequence
    Python loops -- this is the dominant cost (the per-frame for-loop inside
    the fuse function), so batching it cuts wall-clock time by ~G x.
    """
    from mobileposer.synthetic_slam import synthesize_slam_position
    fuse_fn = fuse_fn or sliding_mode_fuse
    device = device or (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))

    lam_vals = torch.tensor([l for l in lam_grid for _ in alpha_grid], dtype=torch.float32, device=device)
    alpha_vals = torch.tensor([a for _ in lam_grid for a in alpha_grid], dtype=torch.float32, device=device)
    G = lam_vals.shape[0]

    err_sum = torch.zeros(G, device=device)
    n = 0
    for subset, source, tran_pred, tran_gt in train_seqs:
        slam_pos = synthesize_slam_position(
            tran_gt, fps=fps, target_rms_m=slam_rms_m,
            seed=(hash(source) % 100000) + seed_base)
        imu_b = tran_pred.to(device).unsqueeze(0).expand(G, -1, -1)
        slam_b = slam_pos.to(device).unsqueeze(0).expand(G, -1, -1)
        gt_b = tran_gt.to(device).unsqueeze(0).expand(G, -1, -1)
        fused = fuse_fn(imu_b, slam_b, lam=lam_vals, alpha=alpha_vals, fps=fps)   # [G,T,3]
        err_sum += (fused - gt_b).norm(dim=-1).mean(dim=-1)
        n += 1
    mean_err = err_sum / max(n, 1)
    best_idx = int(torch.argmin(mean_err).item())
    return float(lam_vals[best_idx].item()), float(alpha_vals[best_idx].item()), float(mean_err[best_idx].item())
