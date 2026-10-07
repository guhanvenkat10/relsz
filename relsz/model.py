"""RelSz inference pipeline.

EDF (SzCORE standard: 19 unipolar channels or 18 double-banana bipolar channels, 256 Hz)
  -> 18 bipolar channels -> causal 0.5-70 Hz band-pass -> per-channel log-spectrogram (2 s window, 1 s step, 40 bands)
  -> minus the trailing 5-min median of the same channel and band (causal)
  -> six networks, probabilities averaged: three channel-set networks (net-s2L) and three that also read
     the left-minus-right difference of the 8 homologous channel pairs (net-s2Las)
  -> the first 60 s of a file raise no alarm (warm-up: the baseline does not exist yet)
  -> alarm when the 3 s trailing mean of the probability is >= 0.4 for 5 consecutive seconds (no back-dating).
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy.signal import butter, sosfilt

FS, WIN, NB, RF = 256, 512, 40, 27
E19 = ("Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6")
DBANANA = ("Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fz-Cz", "Cz-Pz",
           "Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2")
LEFT, RIGHT = list(range(0, 8)), list(range(10, 18))  # left and right chains in DBANANA order
OP = dict(threshold=0.4, smooth=3, consecutive=5)  # alarm rule selected on cross-validated Siena scores
WARMUP_S = 60  # no alarm in a file's first minute; the trailing baseline does not exist yet


def bipolar_matrix():
    M = np.zeros((len(DBANANA), len(E19)), np.float32)
    for i, p in enumerate(DBANANA):
        a, b = p.split("-")
        M[i, E19.index(a)], M[i, E19.index(b)] = 1, -1
    return M


def relative(a, minutes=5, block=10):
    """a[t] minus the median of the (up to) `minutes` of completed `block`-second block means ending at or before t."""
    n, F = a.shape
    nb = n // block
    bm = a[: nb * block].reshape(nb, block, F).mean(1)
    k = minutes * 60 // block
    ref_b = np.empty((nb, F), a.dtype)
    for j in range(nb):
        ref_b[j] = np.median(bm[max(0, j - k + 1): j + 1], axis=0)
    t = np.arange(n)
    j = (t + 1) // block - 1
    ref = np.where((j >= 0)[:, None], ref_b[np.clip(j, 0, max(nb - 1, 0))] if nb else a[:1], a[:1])
    return a - ref


def relspec(x, chunk=3600):
    """x: (C, T) bipolar uV at 256 Hz -> (n_sec, C, 40) relative log-spectrogram; all-zero channels stay all-zero."""
    sos = butter(4, [0.5, 70], btype="band", fs=FS, output="sos")
    valid = np.abs(x).sum(1) > 0
    x = sosfilt(sos, x.astype(np.float64), axis=1).astype(np.float32)
    C, T = x.shape
    n = T // FS
    pad = np.concatenate([np.zeros((C, WIN - FS), np.float32), x[:, : n * FS]], 1)
    xt = torch.from_numpy(pad)
    hann = torch.hann_window(WIN, periodic=False)
    f = torch.fft.rfftfreq(WIN, 1 / FS)
    keep = (f >= 0.5) & (f < 0.5 + NB)
    band = torch.clamp(((f - 0.5) // 1).long(), -1, NB)
    M = torch.zeros(NB, len(f))
    M[band[keep], torch.nonzero(keep).squeeze(1)] = 1
    M = M / M.sum(1, keepdim=True)
    out = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        w = xt[:, s * FS: (e - 1) * FS + WIN].unfold(1, WIN, FS)
        w = w - w.mean(-1, keepdim=True)
        P = torch.fft.rfft(w * hann, dim=-1).abs() ** 2
        out.append(torch.log(P @ M.T + 1e-3).permute(1, 0, 2).numpy())
    L = np.concatenate(out, 0)
    R = relative(L.reshape(n, C * NB)).reshape(n, C, NB)
    R[:, ~valid] = 0
    return np.clip(R, -20, 20).astype(np.float16)


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, dil):
        super().__init__()
        self.pad = (k - 1) * dil
        self.conv = nn.Conv1d(cin, cout, k, dilation=dil)

    def forward(self, x):
        return self.conv(Fn.pad(x, (self.pad, 0)))


class Block(nn.Module):
    def __init__(self, d, k, dil):
        super().__init__()
        self.c1, self.c2 = CausalConv(d, d, k, dil), CausalConv(d, d, 1, 1)
        self.n = nn.LayerNorm(d)

    def forward(self, x):
        return x + self.c2(Fn.gelu(self.n(self.c1(x).transpose(1, 2)).transpose(1, 2)))


class Net(nn.Module):
    def __init__(self, nf=40, d=48):
        super().__init__()
        self.freq = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2), nn.GELU(), nn.Conv1d(16, 32, 5, stride=2, padding=2),
                                  nn.GELU(), nn.Conv1d(32, 32, 3, stride=2, padding=1), nn.GELU(), nn.Flatten(),
                                  nn.Linear(32 * (nf // 4), d))
        self.chan = nn.Sequential(Block(d, 5, 1), Block(d, 5, 2))
        self.mix = nn.Conv1d(2 * d, d, 1)
        self.pool = nn.Sequential(Block(d, 3, 1), Block(d, 3, 2), Block(d, 3, 4))
        self.head = nn.Conv1d(d, 1, 1)

    def forward(self, x, valid):
        B, T, C, F = x.shape
        e = self.freq(x.reshape(B * T * C, 1, F)).reshape(B, T, C, -1)
        e = self.chan(e.permute(0, 2, 3, 1).reshape(B * C, -1, T)).reshape(B, C, -1, T)
        m = valid[:, :, None, None].to(e.dtype)
        mean = (e * m).sum(1) / m.sum(1).clamp(min=1)
        mx = e.masked_fill(m == 0, -1e4).max(1).values
        mx = torch.where(m.sum(1) > 0, mx, torch.zeros_like(mx))
        return self.head(self.pool(self.mix(torch.cat([mean, mx], 1)))).squeeze(1)


def asym(x):
    """x: (..., T, 18, F) -> (..., T, 16, F): left-minus-right and right-minus-left for the 8 homologous pairs; a pair
    is zero if either member is absent (all-zero) in the window."""
    a, b = x[..., LEFT, :], x[..., RIGHT, :]
    ok = ((a.abs().sum((-3, -1), keepdim=True) > 0) & (b.abs().sum((-3, -1), keepdim=True) > 0)).to(x.dtype)
    d = (a - b) * ok
    return torch.cat([d, -d], -2)


class NetA(Net):
    """Net plus a second branch (own frequency encoder and temporal layers) over the asymmetry pseudo-channels."""

    def __init__(self, nf=40, d=48):
        super().__init__(nf, d)
        self.freq_a = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2), nn.GELU(), nn.Conv1d(16, 32, 5, stride=2, padding=2),
                                    nn.GELU(), nn.Conv1d(32, 32, 3, stride=2, padding=1), nn.GELU(), nn.Flatten(),
                                    nn.Linear(32 * (nf // 4), d))
        self.chan_a = nn.Sequential(Block(d, 5, 1), Block(d, 5, 2))
        self.mix = nn.Conv1d(4 * d, d, 1)

    def pooled(self, x, valid, freq, chan):
        B, T, C, F = x.shape
        e = freq(x.reshape(B * T * C, 1, F)).reshape(B, T, C, -1).permute(0, 2, 3, 1).reshape(B * C, -1, T)
        e = chan(e).reshape(B, C, -1, T)
        m = valid[:, :, None, None].to(e.dtype)
        mean = (e * m).sum(1) / m.sum(1).clamp(min=1)
        mx = e.masked_fill(m == 0, -1e4).max(1).values
        return torch.cat([mean, torch.where(m.sum(1) > 0, mx, torch.zeros_like(mx))], 1)

    def forward(self, x, valid):
        xa = asym(x)
        h = torch.cat([self.pooled(x, valid, self.freq, self.chan),
                       self.pooled(xa, xa.abs().sum((1, 3)) > 0, self.freq_a, self.chan_a)], 1)
        return self.head(self.pool(self.mix(h))).squeeze(1)


_models = None


def load_models():
    global _models
    if _models is None:
        d = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
        _models = []
        for tag, cls in (("net-s2L", Net), ("net-s2Las", NetA)):
            for s in (0, 1, 2):
                m = cls()
                m.load_state_dict(torch.load(os.path.join(d, f"{tag}_s{s}.pt"), map_location="cpu"))
                _models.append(m.eval())
    return _models


@torch.no_grad()
def score(spec, chunk=4096):
    valid = torch.from_numpy(np.abs(spec).sum((0, 2)) > 0)[None]
    models = load_models()
    out = np.zeros(len(spec), np.float32)
    for m in models:
        probs = []
        for a in range(0, len(spec), chunk):
            lo = max(0, a - RF)
            x = torch.from_numpy(np.asarray(spec[lo: a + chunk], dtype=np.float32))[None]
            probs.append(torch.sigmoid(m(x, valid))[0, a - lo:].numpy())
        out += np.concatenate(probs) / len(models)
    return out


def causal_mean(p, k):
    """Mean of the last `k` seconds ending at t (fewer at the start of the file)."""
    c = np.cumsum(np.concatenate([[0.0], p.astype(np.float64)]))
    t = np.arange(1, len(p) + 1)
    lo = np.maximum(0, t - k)
    return ((c[t] - c[lo]) / (t - lo)).astype(np.float32)


def alarm_mask(p, threshold=0.4, smooth=3, consecutive=5):
    """Alarm turns on at second t once the `smooth`-second trailing mean of p has been >= threshold for `consecutive`
    seconds ending at t, and stays on while it is >= threshold. Uses only data up to t (no back-dating)."""
    p = causal_mean(p, smooth)
    m = np.zeros(len(p), np.uint8)
    run, state = 0, False
    for i, v in enumerate(p >= threshold):
        run = run + 1 if v else 0
        if not state and run >= consecutive:
            state = True
        elif state and not v:
            state = False
        m[i] = state
    return m


def detect(data, montage_is_bipolar):
    """data: (19, T) unipolar in E19 order, or (18, T) bipolar in DBANANA order; 256 Hz. Returns a 1 Hz alarm mask."""
    x = np.asarray(data, dtype=np.float32)
    if not montage_is_bipolar:
        x = bipolar_matrix() @ x
    p = score(relspec(x))
    p[:WARMUP_S] = 0
    return alarm_mask(p, **OP)
