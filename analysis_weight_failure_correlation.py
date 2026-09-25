"""Does the learned CameraFusionGRU trust-weight shift toward the camera
specifically when/where the IMU is currently wrong? Pooled + per-sequence
correlation between weight_mean_axis (1=trust IMU) and imu_err_m, from
train_camera_fusion.py's per_frame_detail.pt dump.

Run from base_mobileposer/ (or anywhere, path is absolute):
    python analysis_weight_failure_correlation.py runtime_camera_fusion_v2/per_frame_detail.pt
"""
import sys
import numpy as np
import torch
from scipy import stats


def main():
    path = sys.argv[1]
    data = torch.load(path, map_location='cpu', weights_only=False)
    print(f'{len(data)} holdout sequences')

    all_w, all_e = [], []
    per_seq_rho = []
    for rec in data:
        w = rec['weight_mean_axis']
        e = rec['imu_err_m']
        all_w.append(w)
        all_e.append(e)
        if len(w) >= 10 and np.std(w) > 1e-6:
            rho, _ = stats.spearmanr(w, e)
            if not np.isnan(rho):
                per_seq_rho.append(rho)

    all_w = np.concatenate(all_w)
    all_e = np.concatenate(all_e)
    print(f'Total frames: {len(all_w)}')
    print(f'weight (trust-IMU) stats: mean={all_w.mean():.3f} std={all_w.std():.3f} '
          f'min={all_w.min():.3f} max={all_w.max():.3f}')
    print(f'imu_err_m stats: mean={all_e.mean():.3f} std={all_e.std():.3f} '
          f'p90={np.percentile(all_e,90):.3f} max={all_e.max():.3f}')

    rho_pooled, p_pooled = stats.spearmanr(all_w, all_e)
    print(f'\nPooled Spearman(weight, imu_err) = {rho_pooled:.4f}  (p={p_pooled:.3e})')
    print('  Hypothesis: rho < 0 (higher IMU error -> lower trust-IMU weight, i.e. shifts to camera).')

    per_seq_rho = np.array(per_seq_rho)
    print(f'\nPer-sequence Spearman rho: n={len(per_seq_rho)}  mean={per_seq_rho.mean():.4f}  '
          f'median={np.median(per_seq_rho):.4f}')
    frac_negative = float((per_seq_rho < 0).mean())
    print(f'Fraction of sequences with negative rho (weight drops when IMU worse): {frac_negative:.2f}')
    t_stat, p_ttest = stats.wilcoxon(per_seq_rho)
    print(f'Wilcoxon signed-rank on per-sequence rho (vs 0): W={t_stat:.1f}  p={p_ttest:.3e}')

    # quartile comparison: weight in low-IMU-error vs high-IMU-error frames
    q1, q3 = np.percentile(all_e, [25, 75])
    w_low = all_w[all_e <= q1]
    w_high = all_w[all_e >= q3]
    u, p_u = stats.mannwhitneyu(w_low, w_high, alternative='greater')
    print(f'\nWeight when IMU error is LOW  (<=P25={q1:.3f}m): mean={w_low.mean():.3f}  n={len(w_low)}')
    print(f'Weight when IMU error is HIGH (>=P75={q3:.3f}m): mean={w_high.mean():.3f}  n={len(w_high)}')
    print(f'Mann-Whitney (low-err weight > high-err weight): U={u:.1f}  p={p_u:.3e}')


if __name__ == '__main__':
    main()
