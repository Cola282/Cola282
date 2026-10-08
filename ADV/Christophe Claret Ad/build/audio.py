"""Original music + sound design for the Christophe Claret spec ad, all in code.

Music idea: a minute repeater turned into a score. Two "hammer" bells (hours /
quarters), a soft escapement tick, a low pad in D minor that resolves to a D major
chime chord on the logo and rings under the end card.

Outputs (48 kHz, 24-bit WAV) in <out_dir>:
  stems/vo.wav, stems/music.wav, stems/sfx_<category>.wav, stems/sfx.wav
  mix_premaster.wav
Run: python3 -I audio.py <ad_dir> <out_dir>
"""
import json, os, sys
import numpy as np
from scipy import signal
import wave

AD, OUT = sys.argv[1], sys.argv[2]
SR = 48000
TL = json.load(open(os.path.join(AD, "build/timeline.json")))
DUR = TL["duration"]
N = int(DUR * SR)
rng = np.random.default_rng(1989)  # founding year as the seed
os.makedirs(os.path.join(OUT, "stems"), exist_ok=True)


# ---------- helpers ----------
def db(x):
    return 10 ** (x / 20)


def read_wav(p):
    w = wave.open(p)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    if w.getnchannels() == 2:
        x = x.reshape(-1, 2).mean(1)
    assert w.getframerate() == SR
    return x


def write_wav(p, x):
    x = np.atleast_2d(x)
    if x.shape[0] != 2:
        x = np.vstack([x, x]) if x.shape[0] == 1 else x.T
    y = np.clip(x.T, -1, 1)
    y = (y * 8388607).astype(np.int32)
    b = bytearray()
    raw = y.astype("<i4").tobytes()
    # pack 32 -> 24 bit
    a = np.frombuffer(raw, np.uint8).reshape(-1, 4)[:, :3]
    with wave.open(p, "wb") as w:
        w.setnchannels(2); w.setsampwidth(3); w.setframerate(SR)
        w.writeframes(a.tobytes())


def place(buf, x, t, gain=1.0, pan=0.0):
    """Add mono x into stereo buf at time t with equal-power pan (-1..1)."""
    i = int(t * SR)
    if i >= buf.shape[1]:
        return
    x = x[: buf.shape[1] - i]
    l = np.cos((pan + 1) * np.pi / 4); r = np.sin((pan + 1) * np.pi / 4)
    buf[0, i:i + len(x)] += x * gain * l * 1.414
    buf[1, i:i + len(x)] += x * gain * r * 1.414


def env_exp(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def adsr(n, a, r):
    e = np.ones(n)
    na, nr = int(a * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na) if na else 1
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, btype="high", fs=SR, output="sos"), x)


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, btype="low", fs=SR, output="sos"), x)


def noise(n):
    return rng.standard_normal(n).astype(np.float32)


def midi(m):
    return 440 * 2 ** ((m - 69) / 12)


def bell(f, dur=4.0, bright=1.0):
    """Additive bell / gong partials (cathedral-gong flavour)."""
    n = int(dur * SR); t = np.arange(n) / SR
    parts = [(0.5, 1.0, 1.2), (1.0, 1.0, 1.0), (1.19, 0.5, 0.7), (1.56, 0.35, 0.55),
             (2.0, 0.45, 0.45), (2.51, 0.25 * bright, 0.3), (3.01, 0.16 * bright, 0.22),
             (4.07, 0.1 * bright, 0.15)]
    y = np.zeros(n)
    for ratio, amp, tau in parts:
        det = 1 + rng.uniform(-0.0015, 0.0015)
        y += amp * np.sin(2 * np.pi * f * ratio * det * t + rng.uniform(0, 6.28)) * np.exp(-t / (tau * dur / 2.2))
    strike = hp(noise(int(0.012 * SR)), 2500) * env_exp(int(0.012 * SR), 0.003) * 0.35
    y[: len(strike)] += strike
    return (y / np.max(np.abs(y))).astype(np.float32)


def pad_voice(f, n, cutoff=1200):
    t = np.arange(n) / SR
    y = np.zeros(n)
    for d in (-0.06, 0.0, 0.07):  # slightly detuned saws in cents*~
        ff = f * 2 ** (d / 12)
        y += signal.sawtooth(2 * np.pi * ff * t + rng.uniform(0, 6.28))
    return lp(y / 3, cutoff, 2).astype(np.float32)


# ---------- VO ----------
vo_raw = read_wav(os.path.join(AD, TL["vo_file"]))
vo = np.zeros((2, N), np.float32)
place(vo, hp(vo_raw, 70), TL["vo_offset"], gain=1.0)
vo *= db(-1.5)

# speech activity mask (from phrase list) for ducking + SFX dynamic EQ
act = np.zeros(N, np.float32)
for p in TL["phrases"]:
    a, b = int((p["t0"] - 0.04) * SR), int((p["t1"] + 0.06) * SR)
    act[a:b] = 1


def smooth_mask(m, att=0.04, rel=0.25):
    out = np.zeros_like(m); v = 0.0
    ka, kr = 1 - np.exp(-1 / (att * SR)), 1 - np.exp(-1 / (rel * SR))
    # process at control rate for speed
    step = 48
    for i in range(0, len(m), step):
        tgt = m[i]
        v += (tgt - v) * (1 - (1 - (ka if tgt > v else kr)) ** step)
        out[i:i + step] = v
    return out


duck = smooth_mask(act)

# ---------- MUSIC ----------
mus = np.zeros((2, N), np.float32)
beat = 60 / 96.0
t_logo = 19.20
t_drop = 15.98

# 1) escapement tick: soft, every beat, 0.1 -> drop
tick = hp(noise(int(0.02 * SR)), 3000) * env_exp(int(0.02 * SR), 0.0025)
tock = bp(noise(int(0.03 * SR)), 900, 2500) * env_exp(int(0.03 * SR), 0.004)
t = 0.1; k = 0
while t < t_drop - 0.1:
    place(mus, tick if k % 2 == 0 else tock, t, gain=db(-27), pan=-0.15 if k % 2 == 0 else 0.15)
    t += beat / 2; k += 1

# 2) pad in D minor, swelling, opens up in the gap 12.0-13.0, cut at drop
pad = np.zeros(N, np.float32)
for m in (50, 53, 57, 60):  # D3 F3 A3 C4
    pad += pad_voice(midi(m), N, cutoff=900)
pad_env = np.interp(np.arange(N) / SR, [0, 0.6, 4, 11.6, 12.2, 13.0, 15.6, t_drop, DUR],
                    [0, 0.25, 0.45, 0.6, 1.0, 0.8, 0.7, 0.0, 0.0])
pad *= pad_env
lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * np.arange(N) / SR)
mus[0] += pad * db(-30) * (0.8 + 0.2 * lfo)
mus[1] += pad * db(-30) * (0.8 + 0.2 * (1 - lfo))

# 3) hammer bells: minute-repeater figure (low = hours, high = quarters)
low_f, hi_f = midi(50), midi(74)  # D3, D5
pattern = [(0.62, low_f), (2.80, low_f), (4.95, hi_f), (5.55, midi(77)), (6.30, low_f),
           (8.35, hi_f), (8.95, midi(77)), (10.0, midi(81)), (10.95, midi(77)), (11.65, hi_f)]
for tt, f in pattern:
    place(mus, bell(f, 3.0 if f < 200 else 2.2), tt, gain=db(-20 if f < 200 else -24), pan=rng.uniform(-0.4, 0.4))
# cascade in the gap (no VO 12.03-13.0)
for i, m in enumerate([86, 84, 81, 77, 74, 72, 69, 65]):
    place(mus, bell(midi(m), 1.6, 1.2), 12.05 + i * 0.105, gain=db(-17 - i * 0.5), pan=(-1) ** i * 0.5)
# sub swell into the counter
n = int(1.1 * SR); tt = np.arange(n) / SR
sub = np.sin(2 * np.pi * 36.7 * tt) * np.sin(np.pi * tt / 1.1) ** 2
place(mus, sub.astype(np.float32), 12.0, gain=db(-14))
# calibres section: pulsing low bells
for i in range(5):
    place(mus, bell(midi(45), 2.0, 0.6), 13.0 + i * beat * 0.999, gain=db(-24))

# 4) the drop: one sustained tone with slow shimmer
n = int((t_logo - t_drop + 0.4) * SR); tt = np.arange(n) / SR
sus = (np.sin(2 * np.pi * midi(69) * tt) + 0.3 * np.sin(2 * np.pi * midi(81) * tt * 1.001)) * adsr(n, 0.5, 0.4)
sus *= 0.75 + 0.25 * np.sin(2 * np.pi * 3.2 * tt)
place(mus, sus.astype(np.float32), t_drop, gain=db(-24))

# 5) logo: D major chime chord + warm pad, rings out under the end card
for i, m in enumerate([50, 57, 62, 66, 69, 74, 78]):  # D3 A3 D4 F#4 A4 D5 F#5
    place(mus, bell(midi(m), 5.2, 0.9), t_logo + i * 0.018, gain=db(-15 - i * 0.6), pan=(i - 3) * 0.18)
n = N - int(t_logo * SR)
padM = sum(pad_voice(midi(m), n, cutoff=1400) for m in (50, 54, 57, 62))
padM *= np.interp(np.arange(n) / SR, [0, 0.35, 3.0, n / SR - 0.6, n / SR], [0, 1, 0.75, 0.55, 0])
place(mus, padM.astype(np.float32), t_logo, gain=db(-27))
# end-card sparkle (last chord still ringing)
for i, m in enumerate([86, 90, 93]):
    place(mus, bell(midi(m), 1.8, 1.3), 22.62 + i * 0.07, gain=db(-27), pan=0.3 * (i - 1))

# light room reverb on music
def reverb(x, rt=1.6, mix=0.22):
    n = int(rt * SR)
    ir = noise(n) * np.exp(-np.arange(n) / (rt / 6.9 * SR))
    ir = lp(ir, 6000)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = np.vstack([signal.fftconvolve(x[0], ir)[: x.shape[1]], signal.fftconvolve(x[1], ir[::-1])[: x.shape[1]]])
    return x * (1 - mix) + wet * mix * 1.2


mus = reverb(mus)
# ducking: music sits >= 15 dB under the VO while words are spoken
mus *= (1 - duck * (1 - db(-18)))
mus = hp(mus, 35)


# level-aware sidechain: inside speech, force music >= 16 dB under the VO (50 ms windows)
def short_db(x, w=2400):
    e = np.convolve(np.mean(x ** 2, 0), np.ones(w) / w, mode="same")
    return 10 * np.log10(e + 1e-12)


need = short_db(vo, 9600) - 16.5 - short_db(mus, 9600)
g = np.where(act > 0.5, np.minimum(0, need), 0)
# hold the gain reduction 120 ms ahead/behind and smooth it so it never pumps
from scipy.ndimage import minimum_filter1d, uniform_filter1d
g = minimum_filter1d(g, size=int(0.24 * SR))
g = uniform_filter1d(g, size=int(0.08 * SR))
mus *= db(g)

# ---------- SFX ----------
cats = {k: np.zeros((2, N), np.float32) for k in ["whoosh", "hit", "product", "ui", "logo"]}
E = TL["events"]


def whoosh(d=0.45, f0=300, f1=4000, rev=False):
    n = int(d * SR); x = noise(n)
    tt = np.linspace(0, 1, n)
    fc = f0 * (f1 / f0) ** (tt if not rev else 1 - tt)
    y = np.zeros(n); blk = 512
    for i in range(0, n, blk):  # moving band-pass
        c = fc[i]
        y[i:i + blk] = bp(x[i:i + blk + 0], max(60, c * 0.5), min(20000, c * 1.6), 1)[:len(y[i:i + blk])]
    shape = np.sin(np.pi * tt) ** 1.5 if not rev else tt ** 2.5 * (tt < 0.97) + (tt >= 0.97) * (1 - (tt - 0.97) / 0.03)
    y = lp(y * shape, 9000)
    return (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def hit(size=1.0):
    n = int((0.5 + 0.8 * size) * SR); tt = np.arange(n) / SR
    f = 48 + 70 * np.exp(-tt * 35)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / (0.12 + 0.18 * size))
    click = hp(noise(n), 1800) * np.exp(-tt / 0.006) * 0.6
    metal = sum(np.sin(2 * np.pi * fr * tt) * np.exp(-tt / (0.25 * size)) for fr in (1210, 1873, 2640)) * 0.05 * size
    y = body + click + metal
    return (y / np.max(np.abs(y))).astype(np.float32)


def tick_ui(f=4200):
    n = int(0.025 * SR); tt = np.arange(n) / SR
    return (np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.004) + hp(noise(n), 5000) * np.exp(-tt / 0.002) * 0.4).astype(np.float32)


def swish(d=0.28):
    n = int(d * SR); tt = np.linspace(0, 1, n)
    y = bp(noise(n), 1500, 7000) * np.sin(np.pi * tt) ** 3
    return (y / np.max(np.abs(y))).astype(np.float32)


for t in E["whoosh"]:
    place(cats["whoosh"], whoosh(0.42, 250, 5000), t - 0.30, gain=db(-17), pan=rng.uniform(-0.5, 0.5))
for t in E["hit_small"]:
    place(cats["hit"], hit(0.45), t, gain=db(-19))
for t in E["hit_big"]:
    place(cats["hit"], hit(1.2), t, gain=db(-11))
for t in E["whip"]:
    place(cats["whoosh"], whoosh(0.32, 600, 9000), t - 0.18, gain=db(-13), pan=0.6)
    place(cats["hit"], hit(0.7), t + 0.12, gain=db(-17))
for t in E["tick"]:
    place(cats["ui"], tick_ui(2600), t, gain=db(-14))
for t in E["sphere_rise"]:
    n = int(1.4 * SR); tt = np.arange(n) / SR
    f = 400 + 900 * tt / 1.4
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.3 + bp(noise(n), 2000, 6000) * 0.4
    y *= np.sin(np.pi * tt / 1.4) ** 2
    place(cats["product"], y.astype(np.float32), t, gain=db(-24))
for a, b in E["magnet_hum"]:
    n = int((b - a) * SR); tt = np.arange(n) / SR
    y = (np.sin(2 * np.pi * 100 * tt) + 0.5 * np.sin(2 * np.pi * 200 * tt) + 0.25 * signal.square(2 * np.pi * 50 * tt)) * 0.3
    y = bp(y + 0.2 * noise(n), 80, 1500) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.7 * tt)) * adsr(n, 0.4, 0.5)
    place(cats["product"], y.astype(np.float32), a, gain=db(-29))
for t in E["petal"]:
    place(cats["product"], swish(), t, gain=db(-20), pan=rng.uniform(-0.6, 0.6))
for i, t in enumerate(E["montage_cut"]):
    place(cats["whoosh"], whoosh(0.18, 900, 8000), t - 0.1, gain=db(-20), pan=(-1) ** i * 0.5)
    place(cats["hit"], hit(0.3), t, gain=db(-21))
for a, b in E["counter_ticks"]:
    t = a; k = 0
    while t < b:
        place(cats["ui"], tick_ui(3000 + 40 * k), t, gain=db(-25), pan=0.3)
        k += 1; t += 0.11 * np.exp(-k * 0.06) + 0.025
for t in E["stamp"]:
    place(cats["hit"], hit(1.0), t, gain=db(-12))
for t in E["reverse_whoosh"]:
    place(cats["whoosh"], whoosh(0.9, 200, 6000, rev=True), t - 0.9, gain=db(-15))
for a, b in E["riser"]:
    n = int((b - a) * SR); tt = np.linspace(0, 1, n)
    f = 200 * (10 ** (tt * 1.3))
    y = 0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR) + bp(noise(n), 1000, 9000) * tt ** 2
    place(cats["logo"], (y * tt ** 1.5).astype(np.float32), a, gain=db(-20))
for t in E["shimmer"]:
    n = int(2.5 * SR); tt = np.arange(n) / SR
    y = sum(np.sin(2 * np.pi * f * tt + rng.uniform(0, 6)) for f in (5274, 6272, 7040, 8372)) * np.exp(-tt / 0.6) * 0.25
    place(cats["logo"], y.astype(np.float32), t, gain=db(-26))
for t in E["endcard_whoosh"]:
    place(cats["logo"], whoosh(0.4, 300, 6000), t - 0.2, gain=db(-18))
for t in E["endcard_hit"]:
    place(cats["logo"], hit(0.5), t, gain=db(-19))

# dynamic EQ on SFX: high-pass 120 Hz (hits keep their sub via a parallel low band
# only outside speech) and a 2-4 kHz dip while words are spoken
for k in cats:
    x = cats[k]
    low = lp(x, 120)
    hi = hp(x, 120)
    band = bp(hi, 2000, 4000)
    hi = hi - band * duck * (1 - db(-6))
    y = hi + low * (1 - 0.5 * duck)
    if k != "logo":
        y = y * (1 - duck * (1 - db(-5)))
    cats[k] = y.astype(np.float32)
sfx = sum(cats.values())

# ---------- MIX ----------
mix = vo + mus + sfx
for k, x in cats.items():
    write_wav(os.path.join(OUT, f"stems/sfx_{k}.wav"), x)
write_wav(os.path.join(OUT, "stems/sfx.wav"), sfx)
write_wav(os.path.join(OUT, "stems/music.wav"), mus)
write_wav(os.path.join(OUT, "stems/vo.wav"), vo)
write_wav(os.path.join(OUT, "mix_premaster.wav"), mix)

# report: music vs VO level during speech
def rms_db(x, m):
    v = x[:, m > 0.5]
    return 10 * np.log10(np.mean(v ** 2) + 1e-12)

print("VO rms in speech  : %.1f dB" % rms_db(vo, act))
print("Music rms in speech: %.1f dB" % rms_db(mus, act))
print("SFX rms in speech : %.1f dB" % rms_db(sfx, act))
# worst case: short-term (400 ms) music vs VO inside each phrase
for p in TL["phrases"]:
    a, b = int(p["t0"] * SR), int(p["t1"] * SR)
    v = 10 * np.log10(np.mean(vo[:, a:b] ** 2)); m = 10 * np.log10(np.mean(mus[:, a:b] ** 2) + 1e-12)
    print("  phrase %2d  VO-music = %.1f dB" % (p["id"], v - m))
print("peak mix          : %.1f dBFS" % (20 * np.log10(np.max(np.abs(mix)))))
