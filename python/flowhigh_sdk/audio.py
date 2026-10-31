import numpy as np
from math import gcd

SR = 48000
N_FFT = 2048
HOP = 480
N_MELS = 256
FMIN = 20.0
FMAX = 24000.0
LOG_FLOOR = 1e-5
MAG_EPS = 1e-9

_MEL = None
_WIN = None


def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3.0, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mel = np.empty_like(f)
    lin = f < min_log_hz
    mel[lin] = f[lin] / f_sp
    mel[~lin] = min_log_mel + np.log(f[~lin] / min_log_hz) / logstep
    return mel


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3.0, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    hz = np.empty_like(m)
    lin = m < min_log_mel
    hz[lin] = m[lin] * f_sp
    hz[~lin] = min_log_hz * np.exp(logstep * (m[~lin] - min_log_mel))
    return hz


def mel_filterbank():
    global _MEL
    if _MEL is None:
        freqs = np.linspace(0.0, SR / 2.0, N_FFT // 2 + 1)
        edges = np.linspace(_hz_to_mel([FMIN])[0], _hz_to_mel([FMAX])[0], N_MELS + 2)
        hz = _mel_to_hz(edges)
        fb = np.zeros((N_MELS, freqs.shape[0]), dtype=np.float32)
        for m in range(N_MELS):
            left, center, right = hz[m], hz[m + 1], hz[m + 2]
            up = (freqs - left) / (center - left)
            down = (right - freqs) / (right - center)
            fb[m] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / max(1e-8, right - left))
        _MEL = fb
    return _MEL


def hann():
    global _WIN
    if _WIN is None:
        _WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float64)
    return _WIN


def log_mel(audio):
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    pad = (N_FFT - HOP) // 2
    x = np.pad(x, (pad, pad), mode="reflect")
    frames = 1 + (x.size - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    spec = np.fft.rfft(x[idx] * hann(), n=N_FFT, axis=1).T
    mag = np.sqrt(spec.real ** 2 + spec.imag ** 2 + MAG_EPS)
    mel = np.log(np.clip(mel_filterbank() @ mag, LOG_FLOOR, None))
    return mel.T[None].astype(np.float32)


def stft(x, center=True, pad_mode="constant"):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if center:
        p = N_FFT // 2
        x = np.pad(x, (p, p), mode=pad_mode)
    frames = 1 + (x.size - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    return np.fft.rfft(x[idx] * hann(), n=N_FFT, axis=1).T


def istft(spec, length=None, center=True):
    spec = np.asarray(spec)
    frames = np.fft.irfft(spec.T, n=N_FFT, axis=1) * hann()
    out_len = N_FFT + HOP * (frames.shape[0] - 1)
    ola = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    w2 = hann() ** 2
    for i in range(frames.shape[0]):
        s = i * HOP
        ola[s:s + N_FFT] += frames[i]
        wsum[s:s + N_FFT] += w2
    ola /= np.where(wsum > 1e-11, wsum, 1.0)
    if center:
        ola = ola[N_FFT // 2:len(ola) - N_FFT // 2]
    if length is not None:
        if ola.size > length:
            ola = ola[:length]
        elif ola.size < length:
            ola = np.pad(ola, (0, length - ola.size))
    return ola.astype(np.float32)


def cutoff_bin(spec):
    energy = np.cumsum(np.abs(spec).sum(axis=-1))
    if energy[-1] <= 0:
        return 0
    limit = energy[-1] * 0.99
    below = np.nonzero(energy < limit)[0]
    return int(below[-1]) + 1 if below.size else 0


def postprocess(pred, src, length):
    ps = stft(pred, center=True, pad_mode="constant")
    ss = stft(src, center=True, pad_mode="constant")
    n = min(ps.shape[-1], ss.shape[-1])
    ps, ss = ps[:, :n], ss[:, :n]
    cr = cutoff_bin(ss)
    merged = ps.copy()
    merged[:cr] = ss[:cr]
    out = istft(merged, length=length, center=True)
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.99
    return out.astype(np.float32)


def resample(x, sr_in, sr_out=SR):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if sr_in == sr_out:
        return x
    g = gcd(int(sr_out), int(sr_in))
    from scipy.signal import resample_poly
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
