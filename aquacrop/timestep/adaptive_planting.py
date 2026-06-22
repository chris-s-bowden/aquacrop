import numpy as np
import pandas as pd

def adaptive_planting_date(nominal_date, weather, crop, init_cond):
    """First day >= nominal_date meeting the crop's planting trigger(s).
    Returns nominal_date unchanged if triggers are disabled, or the
    latest-allowed day if none qualifies within PlantMaxDelay."""
    if not getattr(crop, "PlantTrigger", None):     # disabled -> no change
        return nominal_date

    dates = weather[:, 4]                            # 0 min,1 max,2 precip,3 et0,4 date
    nominal = np.datetime64(pd.Timestamp(nominal_date))
    start = int(np.argmax(dates >= nominal))         # row of the floor date
    max_delay = int(getattr(crop, "PlantMaxDelay", 60))
    last = min(start + max_delay, len(dates) - 1)

    t_thr    = getattr(crop, "PlantTempThr", None)
    t_win    = int(getattr(crop, "PlantTWindow", 5))
    rain_thr = getattr(crop, "PlantRainThr", None)
    rain_win = int(getattr(crop, "PlantRainWindow", t_win))
    sm_thr   = getattr(crop, "PlantSMThr", None)

    tmean = (weather[:, 0] + weather[:, 1]) / 2.0

    for i in range(start, last + 1):
        ok = True
        if t_thr is not None:                        # N consecutive warm days
            lo = i - t_win + 1
            if lo < start or np.any(tmean[lo:i + 1] < t_thr):
                ok = False
        if ok and rain_thr is not None:              # accumulated rain
            lo = max(start, i - rain_win + 1)
            if weather[lo:i + 1, 2].sum() < rain_thr:
                ok = False
        if ok and sm_thr is not None:                # SM not simulated in off-season 
            if np.mean(init_cond.th[:1]) < sm_thr:   # so this is redundant for now
                ok = False
        if ok:
            return pd.Timestamp(dates[i])
    return pd.Timestamp(dates[last])                 # nothing qualified -> latest allowed