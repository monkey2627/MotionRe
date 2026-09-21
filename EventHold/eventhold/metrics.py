"""Offline retention metrics; never select good-entry episodes for primary error."""
import numpy as np
from .holds import intervals


def retention(error_deg, valid, hz, entry_threshold=10., failure_threshold=15., failure_seconds=1.):
    """One annotated hold, one declared region. Does not label 'false standing'.

    error_deg is a per-frame mean joint angular error for that region. The
    primary error includes wrong-entry frames. Invalid observations remain in
    the denominator report and interrupt sustained-failure runs.
    """
    e=np.asarray(error_deg,dtype=float);valid=np.asarray(valid,dtype=bool)
    if e.ndim!=1 or valid.shape!=e.shape or hz<=0 or len(e)<1:
        raise ValueError('Expected nonempty synchronized per-frame errors and valid mask')
    valid=valid & np.isfinite(e)
    if not valid.any():
        return {'status':'no_valid_reference','valid_frames':0,'total_frames':len(e)}
    n=max(1,int(round(hz)))
    def average(x,m):return float(x[m].mean()) if m.any() else None
    entry=average(e[:n],valid[:n]);late=average(e[-n:],valid[-n:])
    failed=(e>failure_threshold)&valid
    sustained=any((b-a)/hz>=failure_seconds for a,b in intervals(failed))
    correct_entry=None if entry is None else entry<entry_threshold
    delta=late-entry if entry is not None and late is not None and len(e)/hz>=5 else None
    return {'status':'evaluated','mean_error_deg':float(e[valid].mean()),
            'valid_frames':int(valid.sum()),'total_frames':len(e),'entry_error_deg':entry,
            'late_error_deg':late,'retention_delta_deg':delta,
            'correct_entry':correct_entry,'sustained_failure':bool(sustained),
            'failure_all':bool(sustained or (correct_entry is False)),
            'retention_failure_given_correct_entry':bool(sustained) if correct_entry else None,
            'false_standing':'not_inferred_from_angle_error'}
