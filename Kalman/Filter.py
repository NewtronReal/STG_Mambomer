import numpy as np

def kalman_filter_1d(obs, predict_steps=0):
    """
    Apply a 1D Kalman Filter to the entire signal for smoothing,
    and extrapolate forward for `predict_steps` to fill missing data.
    Returns the smoothed signal and predictions.
    """
    n = len(obs)
    x = obs[0] if obs[0] != 0 else np.mean(obs[obs != 0])  # initial state
    P = 1.0                           # initial error covariance
    Q = 1e-5                          # process noise (model uncertainty)
    R = np.var(obs[obs != 0]) or 1e-2  # observation noise
    estimates = []

    for z in obs:
        # Prediction
        P += Q

        # Update only if z != 0 (valid measurement)
        if z != 0:
            K = P / (P + R)
            x += K * (z - x)
            P *= (1 - K)
        # Else: keep x as predicted (no update)

        estimates.append(x)

    # Predict forward for `predict_steps` (if needed)
    preds = []
    for _ in range(predict_steps):
        P += Q
        preds.append(x)  # no update, just predict

    return np.array(estimates), np.array(preds)


def fill_drops_and_denoise(data):
    if len(data.shape) == 3:
        T, V, D = data.shape
        iterator = [(None, v, d) for v in range(V) for d in range(D)]
    elif len(data.shape) == 4:
        N, T, V, D = data.shape
        iterator = [(n, v, d) for n in range(N) for v in range(V) for d in range(D)]
    else:
        raise NotImplementedError('Expected data shape to have 3 or 4 dimensions')

    drops = 0

    for item in iterator:
        n, v, d = item
        ts = data[:, v, d] if n is None else data[n, :, v, d]
        t = 1
        while t < T:
            if ts[t] == 0 and ts[t - 1] != 0:
                start_t = t
                while t < T and ts[t] == 0:
                    t += 1
                prev_data = ts[:start_t]
                smooth_prev, preds = kalman_filter_1d(prev_data, t - start_t)
                ts[:start_t] = smooth_prev
                ts[start_t:t] = preds
                drops += 1
            else:
                t += 1

        # Apply smoothing to rest of the sequence (no new drop found)
        ts_smooth, _ = kalman_filter_1d(ts)
        if n is None:
            data[:, v, d] = ts_smooth
        else:
            data[n, :, v, d] = ts_smooth

    print(f"Smoothed all time series and replaced {drops} sudden drops using Kalman filter")
    return data