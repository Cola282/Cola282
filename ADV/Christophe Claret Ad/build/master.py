"""Master to -14 LUFS integrated / -1 dBTP without touching the voice dynamics.

The VO only gets a static gain. Peak control (look-ahead, smoothed gain riding)
is applied to the music + SFX bus only, and only where the summed true peak would
exceed the ceiling. Run: python3 -I master.py <audio_out_dir>
"""
import sys, os, wave
import numpy as np
import pyloudnorm as pyln
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

D = sys.argv[1]
SR = 48000
TARGET, CEIL = -14.0, -1.0
# stem balance (dB) before mastering: the voice leads
BAL = {"vo": 0.0, "music": 0.0, "sfx": -1.0}


def rd(p):
    w = wave.open(p); b = np.frombuffer(w.readframes(w.getnframes()), np.uint8).reshape(-1, 3)
    x = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
    x = np.where(x >= 1 << 23, x - (1 << 24), x) / 8388608
    return x.reshape(-1, 2).T


def wr(p, x):
    y = (np.clip(x, -1, 1).T * 8388607).astype("<i4")
    a = np.frombuffer(y.tobytes(), np.uint8).reshape(-1, 4)[:, :3]
    with wave.open(p, "wb") as w:
        w.setnchannels(2); w.setsampwidth(3); w.setframerate(SR); w.writeframes(a.tobytes())


def tp_env(x):
    """per-sample true-peak estimate (4x oversampled, max over channels)."""
    up = signal.resample_poly(x, 4, 1, axis=1)
    return np.abs(up).max(0).reshape(-1, 4).max(1)


db = lambda v: 10 ** (v / 20)
meter = pyln.Meter(SR)
vo = rd(os.path.join(D, "stems/vo.wav")) * db(BAL["vo"])
bus_parts = {"music": rd(os.path.join(D, "stems/music.wav")) * db(BAL["music"]),
             "sfx": rd(os.path.join(D, "stems/sfx.wav")) * db(BAL["sfx"])}
bus = bus_parts["music"] + bus_parts["sfx"]

def process(G):
    """static gain G, then transient-only VO peak riding and bus peak control."""
    v = vo * G
    pv = tp_env(v)
    gv = np.minimum(1, db(CEIL) * 0.95 / (pv + 1e-9))
    gv = minimum_filter1d(gv, int(0.004 * SR))      # 2 ms look-ahead / hold
    gv = uniform_filter1d(gv, int(0.006 * SR))      # smooth release, no distortion
    gv = np.minimum(gv, 1)
    v = v * gv
    b = bus * G
    pk = tp_env(v + b)
    gb = np.ones_like(pk)
    over = pk > db(CEIL) * 0.97
    gb[over] = np.clip((db(CEIL) * 0.96 - tp_env(v)[over]) / (tp_env(b)[over] + 1e-9), 0.05, 1)
    gb = np.minimum(uniform_filter1d(minimum_filter1d(gb, int(0.006 * SR)), int(0.004 * SR)), 1)
    return v, b * gb, -20 * np.log10(gv.min()), -20 * np.log10(gb.min())


lo, hi = 0.0, 40.0  # dB of make-up gain
for _ in range(18):
    mid = (lo + hi) / 2
    v, b, _, _ = process(db(mid))
    if meter.integrated_loudness((v + b).T) < TARGET: lo = mid
    else: hi = mid
G = db((lo + hi) / 2)
v, b, vgr, bgr = process(G)
final = v + b
I = meter.integrated_loudness(final.T); TP = 20 * np.log10(tp_env(final).max())
print(f"gain +{20*np.log10(G):.1f} dB | VO transient GR max {vgr:.1f} dB | bus GR max {bgr:.1f} dB")
vo_out = v
wr(os.path.join(D, "master.wav"), final)
# stems at master gain (music/SFX before the bus peak control)
wr(os.path.join(D, "stems/vo_master.wav"), vo_out)
for k in ("music", "sfx"):
    wr(os.path.join(D, f"stems/{k}_master.wav"), bus_parts[k] * G)
print(f"MASTER: {I:.2f} LUFS integrated, {TP:.2f} dBTP")
