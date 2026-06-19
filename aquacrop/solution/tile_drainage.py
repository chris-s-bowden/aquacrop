import numpy as np

def tile_drainage(prof, th, drain_depth, drain_target, drain_coeff):
    """Remove water held above `drain_target` (default = field capacity) in
    compartments whose midpoint is at/below `drain_depth`, at fractional rate
    `drain_coeff` per day. Returns updated th and lateral outflow (mm)."""
    thnew = th.copy()
    q = 0.0
    for ii in range(th.shape[0]):
        if prof.zMid[ii] < drain_depth:
            continue
        tgt = prof.th_fc[ii] if drain_target is None else drain_target
        if thnew[ii] > tgt:
            removed = drain_coeff * (thnew[ii] - tgt)
            thnew[ii] -= removed
            q += removed * prof.dz[ii] * 1000.0   # m3/m3 -> mm
    return thnew, q