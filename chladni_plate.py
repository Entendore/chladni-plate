#!/usr/bin/env python
"""
Chladni Plate Simulator v8 — complete edition
=============================================
Sand settles along the nodal lines of a standing wave on a square or
circular plate, driven by -grad(u^2). The plate can be played like an
instrument: load songs (or use the tone generator) and every editable
frequency band of the music drives its own Chladni mode.

Feature history (all included):
  v1  square plate, vectorized sand physics, mouse pouring
  v2  circular plate (Bessel J_m modes + roots), grain inertia & drag,
      tap, frequency sweep, GIF recording, density rendering, trails,
      colormaps, sand color, grain size, QSettings persistence, hotkeys
  v3  WAV playback (8/16/24/32-bit PCM), QAudioSink streaming,
      real-time FFT: loudness -> amplitude, dominant pitch -> mode,
      built-in sine tone generator, drag & drop
  v4  playlist (MP3/OGG/FLAC/WAV via miniaudio/soundfile), beat
      detection (bass onset) kicks the sand, spectrum overlay,
      auto-advance, transport controls
  v5  frequency LAYERS: per-band energy + pitch -> per-band mode;
      weighted superposition on one plate or split grid of plates
  v6  granular bands: 2-24, log/linear spacing, draggable dividers,
      double-click mute, BandEditor with live energy meters
  v7  per-band gain mini-sliders, ISO octave / 1/3-octave presets,
      JSON layout save/load, musical note edge snapping
  v8  Ctrl-drag fine gain (x0.1, delta-based), Flatten gains / All-on,
      per-band (m,n,sign) override via selectable band list

Deps:  pip install PySide6 numpy
Opt.:  pip install miniaudio soundfile imageio pillow
"""

import colorsys
import json
import os
import sys
import wave as _wave
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QElapsedTimer, QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QLineF, QPainter, QPen, QPointF, QRectF
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
    QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QSlider, QVBoxLayout, QWidget,
)

try:
    from PySide6.QtMultimedia import QAudioFormat, QAudioSink
    HAVE_QTMULTIMEDIA = True
except Exception:
    HAVE_QTMULTIMEDIA = False

try:
    import miniaudio as _miniaudio
except Exception:
    _miniaudio = None

try:
    import soundfile as _sf
except Exception:
    _sf = None

try:
    from scipy.special import jn as _scipy_jn
except Exception:
    _scipy_jn = None

try:
    import imageio.v2 as imageio
    HAVE_IMAGEIO = True
except Exception:
    imageio = None
    HAVE_IMAGEIO = False

PI = np.pi
MAX_PARTICLES = 60_000
MAX_TEXTURE_DIM = 1024
VMAX = 0.6
REC_MAX_FRAMES = 600
SR = 48_000
BANDS = 48
FMIN, FMAX = 20.0, 16000.0
GAIN_MAX = 2.0
BAND_COUNTS = (2, 3, 4, 6, 8, 12, 16, 24)
AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".aif", ".aiff")
RIM = np.array([25, 27, 33], dtype=np.uint8)

OCTAVE_CENTERS = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
THIRD_OCTAVE_CENTERS = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
                        315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
                        3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]
NOTE_NAMES = ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#")


def fmt_time(s):
    m, s = divmod(max(0, int(s)), 60)
    return f"{m}:{s:02d}"


def band_color(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.55, 0.92)
    return (int(r * 255), int(g * 255), int(b * 255))


def band_name(lo, hi):
    def f(v):
        return f"{v / 1000:g}k" if v >= 1000 else (f"{v:.0f}" if v >= 100 else f"{v:g}")
    return f"{f(lo)}\u2013{f(hi)}"


def center_name(fc):
    return f"{fc / 1000.0:g}k" if fc >= 1000 else f"{fc:g}"


def preset_edges(centers, frac):
    """ISO-style band edges from nominal center freqs (frac=1 octave, 3 third-oct)."""
    half = 1.0 / (2.0 * frac)
    edges = [max(FMIN, centers[0] * 2.0 ** -half)]
    for fc in centers[1:]:
        e = fc * 2.0 ** -half
        edges.append(min(max(e, edges[-1] * 1.02), FMAX))
    edges.append(min(FMAX, centers[-1] * 2.0 ** half))
    return edges


def snap_note(f):
    """Nearest equal-temperament pitch -> (freq, semitone offset from A4)."""
    n = int(round(12.0 * np.log2(max(f, 1e-6) / 440.0)))
    return 440.0 * 2.0 ** (n / 12.0), n


def note_label(n):
    return f"{NOTE_NAMES[n % 12]}{4 + (n + 9) // 12}"


# --------------------------------------------------------------------------- colormaps
def _build_colormap(stops, n=2048):
    xs = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], dtype=float)
    t = np.linspace(0.0, 1.0, n)
    lut = np.empty((n, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(t, xs, cs[:, c]).astype(np.uint8)
    return lut


COLORMAPS = {
    "ember":  [(0.00, (56, 78, 210)), (0.30, (24, 36, 120)), (0.50, (8, 9, 16)),
               (0.70, (130, 44, 30)), (1.00, (255, 150, 60))],
    "aurora": [(0.00, (40, 200, 130)), (0.30, (18, 90, 90)), (0.50, (8, 10, 16)),
               (0.70, (110, 40, 110)), (1.00, (240, 130, 190))],
    "mono":   [(0.00, (165, 168, 175)), (0.50, (11, 12, 15)), (1.00, (165, 168, 175))],
}
LUTS = {name: _build_colormap(stops) for name, stops in COLORMAPS.items()}

# --------------------------------------------------------------------------- Bessel
def bessel_jm(x, m):
    """J_m(x) and J_m'(x) via scipy, or a stable Miller downward recurrence."""
    if _scipy_jn is not None:
        z = np.asarray(x, dtype=np.float64)
        Jp1 = _scipy_jn(m + 1, z)
        Jm1 = _scipy_jn(m - 1, z) if m >= 1 else -_scipy_jn(1, z)
        return _scipy_jn(m, z), 0.5 * (Jm1 - Jp1)
    z = np.maximum(np.asarray(x, dtype=np.float64), 1e-6)
    M = int(z.max()) + m + 12
    jk, jk1 = np.ones_like(z), np.zeros_like(z)
    cap, S = {}, np.zeros_like(z)
    for k in range(M, -1, -1):
        S += (jk * jk) if k == 0 else (2.0 * jk * jk)
        if k <= m + 2:
            cap[k] = (jk.copy(), jk1.copy())
        if k == 0:
            break
        jprev = (2.0 * k / z) * jk - jk1
        jk1, jk = jk, jprev
    scale = 1.0 / np.sqrt(np.maximum(S, 1e-300))
    Jm = cap[m][0] * scale
    Jp = (-cap[1][0] if m == 0 else 0.5 * (cap[m - 1][0] - cap[m + 1][0])) * scale
    return Jm, Jp


_ROOT_CACHE = {}


def bessel_root(m, n):
    """n-th positive zero of J_m (cached)."""
    key = (m, n)
    if key not in _ROOT_CACHE:
        def f(t):
            return float(bessel_jm(np.array([t]), m)[0][0])
        guess = (n + 0.5 * m - 0.25) * PI
        roots, xa, fa, xb = [], 1e-3, f(1e-3), 1e-3 + 0.2
        while xb < guess + 2 * PI and len(roots) < n:
            fb = f(xb)
            if fa * fb < 0.0:
                lo, hi, flo = xa, xb, fa
                for _ in range(48):
                    mid = 0.5 * (lo + hi)
                    if flo * f(mid) <= 0.0:
                        hi = mid
                    else:
                        lo, flo = mid, f(mid)
                roots.append(0.5 * (lo + hi))
            xa, fa = xb, fb
            xb += 0.2
        _ROOT_CACHE[key] = roots[n - 1] if roots else guess
    return _ROOT_CACHE[key]


def qimage_to_array(img):
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    h, w, bpl = img.height(), img.width(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), dtype=np.uint8, count=bpl * h)
    return buf.reshape(h, bpl)[:, :w * 3].reshape(h, w, 3).copy()


# --------------------------------------------------------------------------- decoding
def _load_wav_stdlib(path):
    with _wave.open(path, "rb") as w:
        nch, sw, sr, nf = (w.getnchannels(), w.getsampwidth(),
                           w.getframerate(), w.getnframes())
        raw = w.readframes(nf)
    if sw == 1:
        data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        data = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        data = (((b[:, 0] << 8) | (b[:, 1] << 16) | (b[:, 2] << 24)) >> 8)
        data = data.astype(np.float32) / 8388608.0
    elif sw == 4:
        data = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sw}")
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return np.clip(data, -1.0, 1.0), sr


def decode_audio(path):
    """Decode any supported file -> (float32 mono [-1,1], sample_rate)."""
    if _miniaudio is not None:
        try:
            d = _miniaudio.decode_file(path, nchannels=1, sample_rate=SR,
                                       output_format=_miniaudio.SampleFormat.S16)
            x = np.frombuffer(d.samples, dtype=np.int16).astype(np.float32) / 32768.0
            return np.clip(x, -1.0, 1.0), SR
        except Exception:
            pass
    if _sf is not None:
        try:
            x, sr = _sf.read(path, dtype="float32", always_2d=True)
            return np.clip(x.mean(axis=1), -1.0, 1.0), int(sr)
        except Exception:
            pass
    return _load_wav_stdlib(path)


def resample_linear(data, sr_in, sr_out):
    if sr_in == sr_out or data.size == 0:
        return data.astype(np.float32)
    n_out = int(round(data.size * sr_out / sr_in))
    x_out = np.arange(n_out, dtype=np.float64) * (sr_in / sr_out)
    return np.interp(x_out, np.arange(data.size, dtype=np.float64),
                     data).astype(np.float32)


# =========================================================================== audio engine
class AudioEngine(QObject):
    analysed = Signal(float, float, bool)
    layersAnalysed = Signal(list)
    beat = Signal(float)
    positionChanged = Signal(float)
    stateChanged = Signal(str)
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.state = "stopped"
        self.loop = False
        self.volume = 0.8
        self.tone_freq = 440.0
        self.pcm16 = None
        self.pos = 0
        self.sink = None
        self.io = None
        self.rms_ema = 0.0
        self.f_ema = None
        self.spec = np.zeros(BANDS, np.float32)
        self._edges = np.geomspace(30.0, 12000.0, BANDS + 1)
        self.band_edges = []
        self._abuf = np.zeros(0, np.float32)
        self._tphase = 0.0
        self._tf = self.tone_freq
        self._song_t = 0.0
        self._last_beat = -10.0
        self._bass_ema = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._feed)
        self.timer.start()

    def set_bands(self, edges):
        self.band_edges = [(float(lo), float(hi)) for lo, hi in edges]

    def load(self, data, sr):
        x = resample_linear(np.asarray(data, np.float32), sr, SR)
        self.pcm16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
        self.set_source("file")
        return self.pcm16.size

    def set_source(self, src):
        self._hard_stop()
        self.source = src
        self.pos = 0
        self._reset_analysis()
        self.state = "stopped"
        self.stateChanged.emit(self.state)

    def _reset_analysis(self):
        self._abuf = np.zeros(0, np.float32)
        self.spec[:] = 0.0
        self.rms_ema = 0.0
        self.f_ema = None
        self._tphase = 0.0
        self._tf = self.tone_freq
        self._song_t = 0.0
        self._last_beat = -10.0
        self._bass_ema = 0.0

    def _hard_stop(self):
        if self.sink is not None:
            try:
                self.sink.stop()
            except Exception:
                pass

    def _ensure_sink(self):
        if not HAVE_QTMULTIMEDIA:
            return False
        if self.sink is None:
            fmt = QAudioFormat()
            fmt.setSampleRate(SR)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            self.sink = QAudioSink(fmt)
            self.sink.setVolume(self.volume)
        self.io = self.sink.start()
        return self.io is not None

    def play(self):
        if self.source == "file" and self.pcm16 is None:
            return False
        if not self._ensure_sink():
            return False
        self.rms_ema = 0.0
        self.state = "playing"
        self.stateChanged.emit(self.state)
        return True

    def pause(self):
        self._hard_stop()
        self.state = "paused"
        self.stateChanged.emit(self.state)

    def stop(self):
        self._hard_stop()
        self.pos = 0
        self._reset_analysis()
        self.state = "stopped"
        self.stateChanged.emit(self.state)
        self.analysed.emit(0.0, 0.0, False)
        self.positionChanged.emit(0.0)

    def toggle(self):
        if self.state == "playing":
            self.pause()
            return True
        if self.source == "file" and self.pcm16 is None:
            return False
        return self.play()

    def toggle_tone(self):
        if self.source == "tone" and self.state == "playing":
            had_file = self.pcm16 is not None
            self.stop()
            self.source = "file" if had_file else None
            self.stateChanged.emit(self.state)
        else:
            self.set_source("tone")
            self.play()

    def seek(self, sample):
        if self.pcm16 is None:
            return
        self.pos = max(0, min(int(sample), self.pcm16.size))
        self._abuf = np.zeros(0, np.float32)
        if self.state == "playing":
            self._hard_stop()
            self._ensure_sink()
        self.positionChanged.emit(self.pos / SR)

    def set_volume(self, v):
        self.volume = float(min(max(v, 0.0), 1.0))
        if self.sink is not None:
            self.sink.setVolume(self.volume)

    def set_tone_freq(self, f):
        self.tone_freq = float(f)

    def _gen_tone(self, n):
        self._tf += (self.tone_freq - self._tf) * 0.25
        w = 2.0 * PI * self._tf / SR
        x = 0.45 * np.sin(w * np.arange(self._tphase, self._tphase + n,
                                        dtype=np.float64))
        self._tphase += n
        return (x * 32767.0).astype(np.int16)

    def _feed(self):
        if self.state != "playing" or self.sink is None or self.io is None:
            return
        free = self.sink.bytesFree()
        if free <= 0:
            return
        n = free // 2
        try:
            if self.source == "tone":
                chunk = self._gen_tone(n)
                self.io.write(chunk.tobytes())
                self._analyse_chunk(chunk.astype(np.float32) / 32768.0)
            elif self.pcm16 is not None:
                if self.pos >= self.pcm16.size:
                    if self.loop:
                        self.pos = 0
                    else:
                        self._hard_stop()
                        self.state = "stopped"
                        self.stateChanged.emit(self.state)
                        self.finished.emit()
                        return
                k = min(n, self.pcm16.size - self.pos)
                chunk = self.pcm16[self.pos:self.pos + k]
                self.io.write(chunk.tobytes())
                self.pos += k
                self._analyse_chunk(chunk.astype(np.float32) / 32768.0)
                self.positionChanged.emit(self.pos / SR)
        except Exception:
            self.pause()

    def _analyse_chunk(self, x):
        self._abuf = np.concatenate([self._abuf, x])[-8192:]
        seg = self._abuf[-4096:]
        if seg.size < 2048:
            return
        self._song_t += seg.size / SR
        rms = float(np.sqrt(np.mean(seg * seg)))
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        fr = np.fft.rfftfreq(seg.size, 1.0 / SR)
        bi = np.flatnonzero((fr >= 40.0) & (fr <= 5000.0))
        j = bi[int(np.argmax(spec[bi]))]
        peak = float(fr[j])
        strong = (spec[j] / (float(np.mean(spec[bi])) + 1e-12) > 4.0) and rms > 0.003
        self.rms_ema = 0.7 * self.rms_ema + 0.3 * rms
        if strong:
            self.f_ema = peak if self.f_ema is None else 0.7 * self.f_ema + 0.3 * peak

        if self.band_edges:
            data = []
            for lo, hi in self.band_edges:
                sel = (fr >= lo) & (fr <= hi)
                if sel.any():
                    ss = spec[sel]
                    data.append((float(ss.mean()), float(fr[sel][int(np.argmax(ss))])))
                else:
                    data.append((0.0, 0.0))
            self.layersAnalysed.emit(data)

        ebin = np.searchsorted(fr, self._edges)
        vals = np.empty(BANDS, np.float32)
        for i in range(BANDS):
            a, b = ebin[i], max(ebin[i + 1], ebin[i] + 1)
            vals[i] = spec[a:b].mean()
        vals = np.sqrt(np.clip(vals / (vals.max() + 1e-9), 0.0, 1.0))
        self.spec = 0.65 * self.spec + 0.35 * vals

        bb = np.flatnonzero((fr >= 25.0) & (fr <= 160.0))
        bass = float(spec[bb].mean())
        frac = bass / (float(spec.mean()) + 1e-9)
        self._bass_ema = 0.85 * self._bass_ema + 0.15 * bass
        if (frac > 1.6 and bass > 1.45 * self._bass_ema
                and self._song_t - self._last_beat > 0.20):
            self._last_beat = self._song_t
            self.beat.emit(min(1.0, bass / (self._bass_ema + 1e-9) - 0.45))

        self.analysed.emit(self.rms_ema, self.f_ema or 0.0, strong)


# =========================================================================== band editor
class BandEditor(QWidget):
    structureEdited = Signal(str)
    gainEdited = Signal()

    FREQ_TICKS = [(50, "50"), (100, "100"), (200, "200"), (500, "500"),
                  (1000, "1k"), (2000, "2k"), (5000, "5k"), (10000, "10k"),
                  (16000, "16k")]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(175)
        self.setMouseTracking(True)
        self.layers = []
        self.snap = False
        self._drag = None
        self._note_lbl = ""
        self._gain_start = 1.0
        self._gain_start_y = 0.0
        self._fine = False

    def _geom(self):
        h = max(1.0, float(self.height()))
        top = 7.0
        cell_h = max(34.0, h * 0.42)
        gain_top = top + cell_h + 6.0
        gain_h = max(24.0, h - gain_top - 16.0)
        return top, cell_h, gain_top, gain_h

    def _f2x(self, f):
        w = max(1.0, float(self.width()))
        t = (np.log(max(f, 1e-3)) - np.log(FMIN)) / (np.log(FMAX) - np.log(FMIN))
        return t * w

    def _x2f(self, x):
        w = max(1.0, float(self.width()))
        t = min(max(float(x) / w, 0.0), 1.0)
        return float(np.exp(np.log(FMIN) + t * (np.log(FMAX) - np.log(FMIN))))

    def _note_xs(self):
        n0 = int(np.ceil(12.0 * np.log2(FMIN / 440.0)))
        n1 = int(np.floor(12.0 * np.log2(FMAX / 440.0)))
        out, last = [], -1e9
        for n in range(n0, n1 + 1):
            x = self._f2x(440.0 * 2.0 ** (n / 12.0))
            if x - last >= 3.0:
                out.append((x, n))
                last = x
        return out

    def set_layers(self, layers):
        self.layers = list(layers)
        self._drag = None
        self._note_lbl = ""
        self._fine = False
        self.update()

    def _divider_at(self, x, y, top, cell_bot):
        if not (top - 5.0 <= y <= cell_bot + 2.0):
            return None
        best, bd = None, 7.0
        for i in range(1, len(self.layers)):
            d = abs(self._f2x(self.layers[i].lo) - x)
            if d < bd:
                best, bd = i, d
        return best

    def _band_at(self, x):
        for i, L in enumerate(self.layers):
            if x <= self._f2x(L.hi):
                return i
        return len(self.layers) - 1

    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(13, 15, 19))
        if not self.layers:
            p.setPen(QColor(90, 96, 106))
            p.drawText(self.rect(), Qt.AlignCenter, "no bands (single-plate mode)")
            return
        top, cell_h, gain_top, gain_h = self._geom()
        cell_bot = top + cell_h
        gain_bot = gain_top + gain_h
        fnt = p.font()
        fnt.setPointSize(7)
        p.setFont(fnt)

        for fv, lab in self.FREQ_TICKS:
            x = self._f2x(fv)
            p.setPen(QColor(38, 42, 50))
            p.drawLine(QLineF(x, top, x, cell_bot))
            p.setPen(QColor(95, 100, 110))
            p.drawText(QRectF(x - 20, gain_bot + 1, 40, 12), Qt.AlignHCenter, lab)

        if self.snap:
            p.setPen(QColor(48, 54, 66))
            for x, _n in self._note_xs():
                p.drawLine(QLineF(x, cell_bot + 2, x, gain_bot))

        one_y = gain_bot - (1.0 / GAIN_MAX) * gain_h
        gain_drag_i = self._drag[1] if (self._drag and self._drag[0] == "gain") else -1
        for li, L in enumerate(self.layers):
            x0, x1 = self._f2x(L.lo), self._f2x(L.hi)
            if x1 - x0 < 1.2:
                continue
            col = QColor(*L.color)
            if L.enabled:
                p.fillRect(QRectF(x0, top, x1 - x0, cell_h),
                           QColor(col.red() // 6, col.green() // 6, col.blue() // 6))
                eh = max(0.0, min(1.0, L.energy)) * cell_h
                c2 = QColor(col)
                c2.setAlpha(205)
                p.fillRect(QRectF(x0, cell_bot - eh, x1 - x0, eh), c2)
                p.setPen(col)
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(x0, top, x1 - x0, cell_h).adjusted(0, 0, -1, -1))
                if x1 - x0 > 27:
                    p.setPen(QColor(min(255, col.red() + 75), min(255, col.green() + 75),
                                    min(255, col.blue() + 75)))
                    p.drawText(QRectF(x0, top, x1 - x0, cell_h).adjusted(2, 1, -2, -1),
                               Qt.AlignHCenter | Qt.AlignTop, L.name)
            else:
                p.setPen(QColor(68, 72, 80))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(x0, top, x1 - x0, cell_h).adjusted(0, 0, -1, -1))
                p.setPen(QColor(120, 124, 132))
                p.drawText(QRectF(x0, top, x1 - x0, cell_h), Qt.AlignCenter, "off")
            gr = QRectF(x0 + 1, gain_top, max(1.0, x1 - x0 - 2), gain_h)
            p.fillRect(gr, QColor(18, 21, 26))
            gh = max(0.0, min(1.0, L.gain / GAIN_MAX)) * gain_h
            gc = QColor(col)
            gc.setAlpha(165 if L.enabled else 60)
            p.fillRect(QRectF(gr.x(), gain_bot - gh, gr.width(), gh), gc)
            p.setPen(QColor(60, 66, 76))
            p.setBrush(Qt.NoBrush)
            p.drawRect(gr.adjusted(0, 0, -1, -1))
            p.setPen(QColor(255, 255, 255, 36))
            p.drawLine(QLineF(gr.x() + 1, one_y, gr.right() - 1, one_y))
            if gr.width() > 26:
                dragging_this = (li == gain_drag_i)
                txt = (f"{L.gain:.2f}\u00d7" if dragging_this
                       else f"{L.gain:.1f}\u00d7")
                p.setPen(QColor(245, 248, 255) if dragging_this
                         else (QColor(205, 210, 220) if L.enabled
                               else QColor(110, 114, 122)))
                p.drawText(gr.adjusted(0, 1, 0, 0), Qt.AlignHCenter | Qt.AlignTop, txt)

        for i in range(1, len(self.layers)):
            x = self._f2x(self.layers[i].lo)
            hot = bool(self._drag and self._drag[0] == "div" and self._drag[1] == i)
            pen = QPen(QColor(235, 240, 250) if hot else QColor(150, 158, 170))
            pen.setWidth(2 if hot else 1)
            p.setPen(pen)
            p.drawLine(QLineF(x, top - 2, x, cell_bot + 1))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(235, 240, 250) if hot else QColor(150, 158, 170))
            p.drawEllipse(QPointF(x, top - 2), 3.0, 3.0)

        pill_txt = None
        pill_x = None
        if self._drag and self._drag[0] == "div" and self._note_lbl:
            pill_txt = self._note_lbl
            pill_x = self._f2x(self.layers[self._drag[1]].lo)
            pw = 96.0
        elif self._drag and self._drag[0] == "gain":
            L = self.layers[self._drag[1]]
            pill_txt = (f"{L.gain:.2f}\u00d7 \u00b7 fine" if self._fine
                        else f"{L.gain:.2f}\u00d7")
            pill_x = 0.5 * (self._f2x(L.lo) + self._f2x(L.hi))
            pw = 78.0
        if pill_txt is not None:
            rx = min(max(pill_x - pw / 2, 1.0), w - pw - 1.0)
            br = QRectF(rx, top + 1, pw, 14)
            p.setBrush(QColor(20, 23, 30, 230))
            p.setPen(QColor(70, 78, 92))
            p.drawRoundedRect(br, 4, 4)
            p.setPen(QColor(235, 240, 250))
            p.drawText(br, Qt.AlignCenter, pill_txt)

    def mousePressEvent(self, e):
        x, y = e.position().x(), e.position().y()
        top, cell_h, gain_top, gain_h = self._geom()
        cell_bot = top + cell_h
        if e.button() == Qt.LeftButton and len(self.layers) > 1:
            i = self._divider_at(x, y, top, cell_bot)
            if i:
                self._drag = ("div", i)
                f = self.layers[i].lo
                self._note_lbl = (f"{snap_note(f)[0]:.1f} Hz" if self.snap
                                  else f"{f:.0f} Hz")
                self.update()
                e.accept()
                return
        if (e.button() == Qt.LeftButton and self.layers
                and gain_top <= y <= gain_top + gain_h):
            i = self._band_at(x)
            self._drag = ("gain", i)
            self._gain_start = self.layers[i].gain
            self._gain_start_y = y
            self._fine = bool(e.modifiers() & Qt.ControlModifier)
            self.update()
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        x, y = e.position().x(), e.position().y()
        top, cell_h, gain_top, gain_h = self._geom()
        cell_bot = top + cell_h
        if self._drag:
            kind, i = self._drag
            if kind == "div" and 1 <= i < len(self.layers):
                lo_b = self.layers[i - 1].lo * 1.08 + 1.0
                hi_b = self.layers[i].hi / 1.08 - 1.0
                f = min(max(self._x2f(x), lo_b), max(hi_b, lo_b + 0.1))
                if self.snap:
                    f, n = snap_note(f)
                    f = min(max(f, lo_b), max(hi_b, lo_b + 0.1))
                    self._note_lbl = f"{note_label(n)}  \u00b7  {f:.1f} Hz"
                else:
                    self._note_lbl = f"{f:.0f} Hz"
                self.layers[i - 1].hi = f
                self.layers[i].lo = f
                self.update()
                self.structureEdited.emit("div")
                e.accept()
                return
            if kind == "gain" and 0 <= i < len(self.layers):
                self._fine = bool(e.modifiers() & Qt.ControlModifier)
                delta = (self._gain_start_y - y) / gain_h * GAIN_MAX
                if self._fine:
                    delta *= 0.1
                self.layers[i].gain = float(min(max(self._gain_start + delta,
                                                    0.0), GAIN_MAX))
                self.update()
                self.gainEdited.emit()
                e.accept()
                return
        if self.layers and gain_top <= y <= gain_top + gain_h:
            self.setCursor(Qt.SizeVerCursor)
        elif len(self.layers) > 1 and self._divider_at(x, y, top, cell_bot):
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._drag:
            kind, _i = self._drag
            self._drag = None
            self._fine = False
            if kind == "div":
                self._note_lbl = ""
                self.update()
                self.structureEdited.emit("div")
            else:
                self.update()
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        x, y = e.position().x(), e.position().y()
        top, cell_h, gain_top, gain_h = self._geom()
        if self.layers and gain_top <= y <= gain_top + gain_h:
            self.layers[self._band_at(x)].gain = 1.0
            self.update()
            self.gainEdited.emit()
            e.accept()
            return
        if y <= top + cell_h:
            for L in self.layers:
                if self._f2x(L.lo) <= x <= self._f2x(L.hi):
                    L.enabled = not L.enabled
                    self.update()
                    self.structureEdited.emit("mute")
                    e.accept()
                    return
        super().mouseDoubleClickEvent(e)


# =========================================================================== layers
@dataclass
class Layer:
    name: str
    lo: float
    hi: float
    color: tuple = (200, 200, 200)
    gain: float = 1.0
    enabled: bool = True
    override: bool = False
    energy: float = 0.0
    peak: float = 0.0
    env: float = 1e-6
    m: int = 2
    n: int = 2
    sign: float = 1.0
    cand: object = field(default=None, repr=False)
    cand_count: int = field(default=0, repr=False)


# =========================================================================== plate engine
class PlateEngine:
    def __init__(self, name="plate", tint=(138, 180, 248)):
        self.name = name
        self.tint = tint
        self.shape = "square"
        self.m, self.n, self.sign = 3, 2, -1.0
        self.super_layers = []
        self._grid_cache = {}
        self.drive_scale = 1.0
        self.flash = 0.0
        self.last_s = 1.0
        self.speed = 1.05
        self.jitter = 8e-4
        self.momentum = True
        self.drag = 0.65
        self.cmap = LUTS["ember"]
        self._sand = np.array([255, 226, 150], dtype=np.float32)
        self.psize = 2
        self._dens_gain = 1.7 / 4.0
        self.trails = False
        self.trail_decay = 0.88
        self.rng = np.random.default_rng()
        self.px = np.zeros(0, np.float32)
        self.py = np.zeros(0, np.float32)
        self.vx = np.zeros(0, np.float32)
        self.vy = np.zeros(0, np.float32)
        self._w = self._h = 0
        self._trail = None
        self.qimage = None

    @staticmethod
    def _ranges(shape):
        return ((1, 12), (1, 12)) if shape == "square" else ((0, 8), (1, 8))

    def set_shape(self, shape):
        if shape != self.shape:
            self.shape = shape
            self._grid_cache.clear()
            (mlo, mhi), (nlo, nhi) = self._ranges(shape)
            self.m, self.n = min(max(self.m, mlo), mhi), min(max(self.n, nlo), nhi)

    def set_mode(self, m=None, n=None, sign=None):
        (mlo, mhi), (nlo, nhi) = self._ranges(self.shape)
        if m is not None:
            self.m = min(max(int(m), mlo), mhi)
        if n is not None:
            self.n = min(max(int(n), nlo), nhi)
        if sign is not None:
            self.sign = 1.0 if sign >= 0 else -1.0

    def set_size(self, w, h):
        w, h = max(2, int(w)), max(2, int(h))
        if w != self._w or h != self._h:
            self._w, self._h = w, h
            self._trail = np.zeros((h, w), np.float32)
            self._grid_cache.clear()

    def set_sand_color(self, r, g, b):
        self._sand = np.array([r, g, b], dtype=np.float32)

    def set_particle_size(self, size):
        self.psize = int(size)
        self._dens_gain = 1.7 / (self.psize * self.psize)

    def set_trails(self, on):
        self.trails = bool(on)
        if not on and self._trail is not None:
            self._trail[:] = 0.0

    def _sample(self, count):
        x, y = self.rng.random(count), self.rng.random(count)
        if self.shape == "circle":
            X, Y = (x - 0.5) * 2.0, (y - 0.5) * 2.0
            r = np.hypot(X, Y)
            f = 0.99 / np.maximum(r, 1e-9)
            X, Y = np.where(r > 0.99, X * f, X), np.where(r > 0.99, Y * f, Y)
            x, y = 0.5 + 0.5 * X, 0.5 + 0.5 * Y
        return x.astype(np.float32), y.astype(np.float32)

    def set_count(self, count):
        count = max(0, min(int(count), MAX_PARTICLES))
        cur = self.px.size
        if count > cur:
            x, y = self._sample(count - cur)
            self.px = np.concatenate([self.px, x])
            self.py = np.concatenate([self.py, y])
            self.vx = np.concatenate([self.vx, np.zeros(count - cur, np.float32)])
            self.vy = np.concatenate([self.vy, np.zeros(count - cur, np.float32)])
        elif count < cur:
            self.px, self.py = self.px[:count].copy(), self.py[:count].copy()
            self.vx, self.vy = self.vx[:count].copy(), self.vy[:count].copy()

    def reset_sand(self, count):
        self.px, self.py = self._sample(max(0, int(count)))
        self.vx = np.zeros(self.px.size, np.float32)
        self.vy = np.zeros(self.px.size, np.float32)

    def beat_pulse(self, strength=1.0):
        s = min(max(strength, 0.0), 1.0)
        if self.px.size:
            if self.momentum:
                k = 0.28 + 0.5 * s
                self.vx += self.rng.normal(0.0, k, self.px.size).astype(np.float32)
                self.vy += self.rng.normal(0.0, k, self.px.size).astype(np.float32)
            else:
                d = 0.005 + 0.012 * s
                self.px += self.rng.normal(0.0, d, self.px.size).astype(np.float32)
                self.py += self.rng.normal(0.0, d, self.py.size).astype(np.float32)
                np.clip(self.px, 0.0, 1.0, out=self.px)
                np.clip(self.py, 0.0, 1.0, out=self.py)
        self.flash = min(1.6, self.flash + 0.7 + 0.8 * s)

    def pour(self, x, y, sigma, k=350):
        if self.px.size >= MAX_PARTICLES:
            return
        nx, ny = self.rng.normal(x, sigma, k), self.rng.normal(y, sigma, k)
        if self.shape == "circle":
            X, Y = (nx - 0.5) * 2.0, (ny - 0.5) * 2.0
            r = np.hypot(X, Y)
            f = 0.99 / np.maximum(r, 1e-9)
            X, Y = np.where(r > 0.99, X * f, X), np.where(r > 0.99, Y * f, Y)
            nx, ny = 0.5 + 0.5 * X, 0.5 + 0.5 * Y
        self.px = np.concatenate([self.px, np.clip(nx, 0, 1).astype(np.float32)])
        self.py = np.concatenate([self.py, np.clip(ny, 0, 1).astype(np.float32)])
        self.vx = np.concatenate([self.vx, np.zeros(k, np.float32)])
        self.vy = np.concatenate([self.vy, np.zeros(k, np.float32)])

    def _layer_grid(self, m, n, sign):
        key = (m, n, round(sign), self.shape, self._w, self._h)
        c = self._grid_cache.get(key)
        if c is not None:
            return c
        w, h = self._w, self._h
        x = np.linspace(0.0, 1.0, w)
        y = np.linspace(0.0, 1.0, h)
        mask = None
        if self.shape == "square":
            u = (np.outer(np.cos(PI * m * y), np.cos(PI * n * x))
                 + sign * np.outer(np.cos(PI * n * y), np.cos(PI * m * x)))
        else:
            k = bessel_root(m, n)
            Xg, Yg = np.meshgrid(x * 2.0 - 1.0, y * 2.0 - 1.0)
            R = np.hypot(Xg, Yg)
            TH = np.arctan2(Yg, Xg)
            rprof = np.linspace(0.0, np.sqrt(2.0), 4096)
            J = np.interp(R, rprof, bessel_jm(rprof * k, m)[0])
            u = J * (np.cos(m * TH) if sign > 0 else np.sin(m * TH))
            mask = R > 1.0
        src = u[~mask] if mask is not None else u
        un = float(np.abs(src).max()) or 1.0
        u = np.where(mask, 0.0, u / un) if mask is not None else u / un
        c = (u.astype(np.float32), un, mask)
        if len(self._grid_cache) > 64:
            self._grid_cache.pop(next(iter(self._grid_cache)))
        self._grid_cache[key] = c
        return c

    def _grad_mode(self, x, y, m, n, sign, inv=1.0):
        if self.shape == "square":
            a, b = PI * n * x, PI * m * y
            c, d = PI * m * x, PI * n * y
            cnx, cmy, cmx, cny = np.cos(a), np.cos(b), np.cos(c), np.cos(d)
            u = (cnx * cmy + sign * cmx * cny) * inv
            ux = (-PI * n * np.sin(a) * cmy - sign * PI * m * np.sin(c) * cny) * inv
            uy = (-PI * m * cnx * np.sin(b) - sign * PI * n * cmx * np.sin(d)) * inv
            return u, ux, uy
        X, Y = x * 2.0 - 1.0, y * 2.0 - 1.0
        r = np.hypot(X, Y)
        rs = np.maximum(r, 1e-4)
        ct, st = X / rs, Y / rs
        k = bessel_root(m, n)
        J, Jp = bessel_jm(k * r, m)
        ang = m * np.arctan2(Y, X)
        if sign > 0:
            C, Cp = np.cos(ang), -m * np.sin(ang)
        else:
            C, Cp = np.sin(ang), m * np.cos(ang)
        A, B = k * Jp * C, J * Cp / rs
        return (J * C * inv, (A * ct - B * st) * inv, (A * st + B * ct) * inv)

    def _active(self):
        sl = [t for t in self.super_layers if t[3] > 0.02]
        if len(sl) > 10:
            sl.sort(key=lambda t: -t[3])
            sl = sl[:10]
        return sl

    def _force(self, x, y):
        if not self.super_layers:
            _, un, _ = self._layer_grid(self.m, self.n, self.sign)
            return self._grad_mode(x, y, self.m, self.n, self.sign,
                                   1.0 / max(un, 1e-9))
        U = Gx = Gy = None
        for (m, n, sg, w) in self._active():
            _, un, _ = self._layer_grid(m, n, sg)
            u, gx, gy = self._grad_mode(x, y, m, n, sg, w / max(un, 1e-9))
            U = u if U is None else U + u
            Gx = gx if Gx is None else Gx + gx
            Gy = gy if Gy is None else Gy + gy
        if U is None:
            z = np.zeros_like(x)
            return z, z, z
        return U, Gx, Gy

    def move(self, dt, phase, amp_eff):
        if self.px.size == 0:
            return
        drive = (amp_eff * phase) ** 2
        u, gx, gy = self._force(self.px, self.py)
        gx *= 2.0 * u
        gy *= 2.0 * u
        g = np.hypot(gx, gy)
        clip = np.minimum(1.0, 60.0 / (g + 1e-9))

        if self.momentum:
            A = 0.9 * self.speed * drive
            ret = 1.0 - self.drag
            self.vx = ret * self.vx - A * gx * clip * dt
            self.vy = ret * self.vy - A * gy * clip * dt
            sp = np.hypot(self.vx, self.vy)
            fast = sp > VMAX
            if fast.any():
                self.vx[fast] *= VMAX / sp[fast]
                self.vy[fast] *= VMAX / sp[fast]
            self.px += self.vx * dt
            self.py += self.vy * dt
        else:
            step = 0.25 * self.speed * drive
            self.px -= step * gx * clip * dt
            self.py -= step * gy * clip * dt

        if self.jitter > 0.0:
            s = self.jitter * drive * dt * 60.0
            self.px += self.rng.normal(0.0, s, self.px.size).astype(np.float32)
            self.py += self.rng.normal(0.0, s, self.py.size).astype(np.float32)

        np.clip(self.px, 0.0, 1.0, out=self.px)
        np.clip(self.py, 0.0, 1.0, out=self.py)
        if self.momentum:
            self.vx[self.px <= 0.0] *= -0.5
            self.vx[self.px >= 1.0] *= -0.5
            self.vy[self.py <= 0.0] *= -0.5
            self.vy[self.py >= 1.0] *= -0.5

        if self.shape == "circle":
            X, Y = (self.px - 0.5) * 2.0, (self.py - 0.5) * 2.0
            r = np.hypot(X, Y)
            bad = r > 0.995
            if bad.any():
                f = 0.995 / r[bad]
                Xb, Yb = X[bad] * f, Y[bad] * f
                self.px[bad], self.py[bad] = 0.5 + 0.5 * Xb, 0.5 + 0.5 * Yb
                if self.momentum:
                    nx, ny = Xb / 0.995, Yb / 0.995
                    vn = self.vx[bad] * nx + self.vy[bad] * ny
                    bump = np.where(vn > 0.0, 1.3 * vn, 0.0)
                    self.vx[bad] -= bump * nx
                    self.vy[bad] -= bump * ny

    def render(self, s):
        self.last_s = s
        w, h = self._w, self._h
        if w < 2 or h < 2:
            return
        NM1 = len(self.cmap) - 1
        mask = None
        if not self.super_layers:
            g, _, mask = self._layer_grid(self.m, self.n, self.sign)
            u = g * s
        else:
            act = self._active()
            u = None
            for (m, n, sg, wt) in act:
                g, _, mk = self._layer_grid(m, n, sg)
                mask = mk if mask is None else mask
                u = g * wt if u is None else u + g * wt
            if u is None:
                u = np.zeros((h, w), np.float32)
            else:
                u *= s
                u /= max(float(np.abs(u).max()), 0.15)
        idx = (u * 0.5 + 0.5) * NM1
        np.clip(idx, 0, NM1, out=idx)
        rgb = self.cmap[idx.astype(np.int32)]
        if mask is not None:
            rgb[mask] = RIM

        if self.px.size:
            P = self.psize
            ix = np.clip((self.px * (w - 1)).astype(np.int32), 0, w - P)
            iy = np.clip((self.py * (h - 1)).astype(np.int32), 0, h - P)
            base = iy.astype(np.int64) * w
            counts = np.zeros(w * h, np.int64)
            for dy in range(P):
                for dx in range(P):
                    counts += np.bincount(base + dy * w + ix + dx, minlength=w * h)
            alpha = counts.reshape(h, w).astype(np.float32) * self._dens_gain
            np.clip(alpha, 0.0, 0.95, out=alpha)
            if self.trails and self._trail is not None:
                self._trail *= self.trail_decay
                self._trail += alpha * (1.0 - self.trail_decay)
                np.clip(self._trail, 0.0, 0.95, out=self._trail)
                alpha = np.clip(alpha * 0.8 + self._trail * 0.55, 0.0, 0.95)
            rows, cols = np.nonzero(alpha > 0.01)
            if rows.size:
                a = alpha[rows, cols][:, None]
                bc = rgb[rows, cols].astype(np.float32)
                rgb[rows, cols] = (bc * (1.0 - a) + self._sand * a).astype(np.uint8)

        self.qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


# =========================================================================== plate widget
class ChladniPlate(QWidget):
    modeChanged = Signal(int, int, float)
    shapeChanged = Signal(str)
    sandCountChanged = Signal(int)
    recordingChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(380, 380)
        self.setMouseTracking(True)

        self.display = "single"
        self.layers = []
        self.manual = (3, 2, -1.0)
        self.shape = "square"

        self.vibrate = True
        self.vib_freq = 0.6
        self.amp = 1.0
        self.audio_reactive = False
        self._audio_drive = 1.0
        self.speed = 1.05
        self.jitter = 8e-4
        self.momentum = True
        self.drag = 0.65
        self.paused = False
        self._phase = 1.0
        self._time = 0.0

        self.cmap = LUTS["ember"]
        self._sand_rgb = (255, 226, 150)
        self.psize = 2
        self.trails = False
        self.trail_decay = 0.88
        self.brush_sigma = 0.036
        self.count_per = 4000
        self.show_spectrum = False
        self._spec = None

        self.engines = [PlateEngine("plate")]
        self._dirty = True
        self._pouring = False
        self._recording = False
        self._rec_frames = []

        self._clock = QElapsedTimer()
        self._clock.start()
        self._last = 0.0
        self._tick_i = 0
        self._fps_n = 0

        self._propagate()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def _propagate(self):
        for e in self.engines:
            e.cmap = self.cmap
            e.set_sand_color(*self._sand_rgb)
            e.set_particle_size(self.psize)
            e.set_trails(self.trails)
            e.trail_decay = self.trail_decay
            e.speed, e.jitter = self.speed, self.jitter
            e.momentum, e.drag = self.momentum, self.drag
            e.set_shape(self.shape)
            e.set_count(self.count_per)

    def _total(self):
        return sum(e.px.size for e in self.engines)

    def _rects(self):
        W, H = float(self.width()), float(self.height())
        n = len(self.engines)
        gap = 6.0 if n > 4 else 8.0
        if n <= 1:
            return [QRectF(0, 0, W, H)]
        if n == 2:
            w = (W - gap) / 2.0
            return [QRectF(0, 0, w, H), QRectF(w + gap, 0, w, H)]
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        w = (W - gap * (cols - 1)) / cols
        h = (H - gap * (rows - 1)) / rows
        return [QRectF((i % cols) * (w + gap), (i // cols) * (h + gap), w, h)
                for i in range(n)]

    def set_display(self, display, layers):
        self.layers = list(layers)
        en = [L for L in self.layers if L.enabled] or self.layers[:1]
        need = (display != self.display
                or len(self.engines) != (len(en) if display == "split" else 1))
        if need:
            old = self.engines[0] if self.engines else None
            self.display = display
            if display == "split":
                self.engines = [PlateEngine(L.name, L.color) for L in en]
            else:
                self.engines = [PlateEngine("superposed" if display == "super"
                                            else "plate")]
            for e in self.engines:
                if old is not None and old.px.size:
                    e.px, e.py = old.px.copy(), old.py.copy()
                    e.vx, e.vy = old.vx.copy(), old.vy.copy()
        else:
            if display == "split":
                for e, L in zip(self.engines, en):
                    e.name, e.tint = L.name, L.color
            else:
                self.engines[0].name = ("superposed" if display == "super" else "plate")
        self._propagate()
        self._dirty = True

    def refresh_layer_metadata(self):
        en = [L for L in self.layers if L.enabled] or self.layers[:1]
        if self.display == "split":
            for e, L in zip(self.engines, en):
                e.name, e.tint = L.name, L.color
        self._dirty = True

    def set_super(self, modes, weights):
        if not self.engines:
            return
        sl = []
        for i, L in enumerate(self.layers):
            if not L.enabled:
                continue
            md = modes[i] if i < len(modes) and modes[i] else (L.m, L.n, L.sign)
            wt = weights[i] if i < len(weights) else 0.0
            sl.append((int(md[0]), int(md[1]), float(md[2]), float(wt)))
        self.engines[0].super_layers = sl
        self._dirty = True

    def set_split_modes(self, items):
        for e, (md, wt) in zip(self.engines, items):
            e.drive_scale = min(2.0, max(0.0, float(wt)))
            if md:
                e.set_mode(m=md[0], n=md[1], sign=md[2])
        self._dirty = True

    def layer_mode(self):
        e = self.engines[0]
        return e.m, e.n, e.sign

    def set_mode(self, m=None, n=None, sign=None):
        (mlo, mhi), (nlo, nhi) = PlateEngine._ranges(self.shape)
        if m is not None:
            self.manual = (min(max(int(m), mlo), mhi), self.manual[1], self.manual[2])
        if n is not None:
            self.manual = (self.manual[0], min(max(int(n), nlo), nhi), self.manual[2])
        if sign is not None:
            self.manual = (self.manual[0], self.manual[1],
                           1.0 if sign >= 0 else -1.0)
        if self.engines:
            self.engines[0].set_mode(m, n, sign)
        self._dirty = True
        e = self.engines[0]
        self.modeChanged.emit(e.m, e.n, e.sign)

    def set_shape(self, shape):
        if shape == self.shape:
            return
        self.shape = shape
        self._propagate()
        self._dirty = True
        self.shapeChanged.emit(shape)
        e = self.engines[0]
        self.manual = (e.m, e.n, self.manual[2])
        self.modeChanged.emit(e.m, e.n, e.sign)

    def randomize_mode(self):
        rng = self.engines[0].rng
        (mlo, mhi), (nlo, nhi) = PlateEngine._ranges(self.shape)
        m = n = 1
        s = 1.0
        for _ in range(60):
            m = int(rng.integers(mlo, mhi + 1))
            n = int(rng.integers(nlo, nhi + 1))
            s = float(rng.choice([-1.0, 1.0]))
            if not (self.shape == "square" and s < 0 and m == n):
                break
        self.set_mode(m, n, s)

    def _sweep_entries(self):
        if self.shape == "square":
            ent = [(m, n, s) for m in range(1, 8) for n in range(1, 8)
                   for s in (1.0, -1.0) if not (s < 0 and m == n)]
            ent.sort(key=lambda e: (e[0] ** 2 + e[1] ** 2, e[0], e[1], e[2]))
            return ent
        ent = [(m, n) for m in range(0, 7) for n in range(1, 7)]
        ent.sort(key=lambda e: bessel_root(e[0], e[1]))
        return ent

    def next_sweep(self):
        ent = self._sweep_entries()
        e = self.engines[0]
        cur = ((e.m, e.n, round(e.sign)) if self.shape == "square" else (e.m, e.n))
        i = ent.index(cur) if cur in ent else -1
        nx = ent[(i + 1) % len(ent)]
        self.set_mode(nx[0], nx[1], nx[2] if self.shape == "square" else None)

    def set_colormap(self, index):
        self.cmap = LUTS[list(COLORMAPS)[index]]
        self._propagate()
        self._dirty = True

    def set_sand_color(self, r, g, b):
        self._sand_rgb = (r, g, b)
        self._propagate()
        self._dirty = True

    def set_particle_size(self, size):
        self.psize = int(size)
        self._propagate()
        self._dirty = True

    def set_trails(self, on):
        self.trails = bool(on)
        self._propagate()
        self._dirty = True

    def set_particle_count(self, count):
        self.count_per = max(0, min(int(count), MAX_PARTICLES))
        for e in self.engines:
            e.set_count(self.count_per)
        self.sandCountChanged.emit(self._total())
        self._dirty = True

    def reset_sand(self, count=None):
        count = self.count_per if count is None else max(0, min(int(count), MAX_PARTICLES))
        self.count_per = count
        for e in self.engines:
            e.reset_sand(count)
        self.sandCountChanged.emit(self._total())
        self._dirty = True

    def tap(self):
        for e in self.engines:
            e.beat_pulse(1.0)

    def beat_pulse(self, strength=1.0):
        for e in self.engines:
            e.beat_pulse(strength)

    def set_audio_drive(self, v):
        self._audio_drive = float(min(max(v, 0.0), 2.0))

    def _eff_amp(self):
        a = self.amp * (self._audio_drive if self.audio_reactive else 1.0)
        return min(max(a, 0.0), 2.0)

    def set_show_spectrum(self, on):
        self.show_spectrum = bool(on)

    def set_spectrum(self, arr):
        self._spec = arr

    def mark_dirty(self):
        self._dirty = True

    def take_fps(self):
        n, self._fps_n = self._fps_n, 0
        return n

    def _compose(self):
        dpr = self.devicePixelRatioF()
        img = QImage(max(2, int(self.width() * dpr)), max(2, int(self.height() * dpr)),
                     QImage.Format.Format_RGB888)
        img.fill(QColor(8, 9, 12))
        p = QPainter(img)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        for e, r in zip(self.engines, self._rects()):
            if e.qimage is not None:
                p.drawImage(QRectF(r.x() * dpr, r.y() * dpr,
                                   r.width() * dpr, r.height() * dpr), e.qimage)
        p.end()
        return img

    def tick(self):
        if not self.isVisible():
            return
        now = self._clock.nsecsElapsed() * 1e-9
        dt = float(np.clip(now - self._last, 0.001, 0.05))
        self._last = now
        dpr = self.devicePixelRatioF()
        for e, r in zip(self.engines, self._rects()):
            e.flash *= float(np.exp(-5.0 * dt))
            shrink = min(1.0, MAX_TEXTURE_DIM / max(1.0, r.width() * dpr),
                         MAX_TEXTURE_DIM / max(1.0, r.height() * dpr))
            e.set_size(r.width() * dpr * shrink, r.height() * dpr * shrink)

        if not self.paused:
            self._time += dt
            self._phase = float(np.cos(2.0 * PI * self.vib_freq * self._time))
            base = self._eff_amp()
            ph = self._phase if self.vibrate else 1.0
            for e in self.engines:
                a = base * (e.drive_scale if self.display == "split" else 1.0)
                e.move(dt, ph, a)
                e.render(a * ph)
            self._dirty = False
        elif self._dirty:
            for e in self.engines:
                e.render(e.last_s)
            self._dirty = False
        self._fps_n += 1
        self.update()

        if self._recording and not self.paused and self._tick_i % 2 == 0 \
                and len(self._rec_frames) < REC_MAX_FRAMES:
            img = self.grab().toImage()
            small = img.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._rec_frames.append(qimage_to_array(small))
            if len(self._rec_frames) >= REC_MAX_FRAMES:
                self.stop_recording()
        self._tick_i += 1

    def start_recording(self):
        if not self._recording:
            self._recording, self._rec_frames = True, []
            self.recordingChanged.emit(True)

    def stop_recording(self):
        if self._recording:
            self._recording = False
            self.recordingChanged.emit(False)

    def pop_recording(self):
        frames, self._rec_frames = self._rec_frames, []
        return frames

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        many = len(self.engines) > 4
        for e, r in zip(self.engines, self._rects()):
            if e.qimage is not None:
                p.drawImage(r, e.qimage)
            else:
                p.fillRect(r, QColor(8, 9, 12))
            p.setRenderHint(QPainter.Antialiasing)
            p.setBrush(Qt.NoBrush)
            p.setPen(QColor(70, 76, 88))
            if self.shape == "square":
                p.drawRect(r.adjusted(-1, -1, 0, 0))
            else:
                p.drawEllipse(r.adjusted(-1, -1, 0, 0))
            f = p.font()
            f.setPointSize(8 if many else (10 if len(self.engines) > 2 else 11))
            p.setFont(f)
            var = ""
            if self.shape == "square" and not e.super_layers:
                var = " +" if e.sign > 0 else " \u2212"
            if e.super_layers:
                txt = f"{e.name} \u00b7 {len(e.super_layers)} layers"
            elif r.width() < 90:
                txt = f"({e.m},{e.n})"
            else:
                txt = f"{e.name}   ({e.m},{e.n}){var}"
            p.setPen(QColor(e.tint[0], e.tint[1], e.tint[2], 220))
            p.drawText(r.adjusted(6, 3, -6, -4), Qt.AlignTop | Qt.AlignLeft, txt)
            if e.flash > 0.03:
                a = int(160 * min(1.0, e.flash))
                p.setPen(QColor(self._sand_rgb[0], self._sand_rgb[1],
                                self._sand_rgb[2], a))
                m = 2 + 8 * (1.0 - min(1.0, e.flash))
                rr = r.adjusted(m, m, -m, -m)
                if self.shape == "circle":
                    p.drawEllipse(rr)
                else:
                    p.drawRect(rr)

        if self.show_spectrum and self._spec is not None:
            w, h = self.width(), self.height()
            bw = w / len(self._spec)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(self._sand_rgb[0], self._sand_rgb[1],
                              self._sand_rgb[2], 150))
            for i, v in enumerate(self._spec):
                bh = max(1.0, float(v) * h * 0.13)
                p.drawRect(QRectF(i * bw + 1.0, h - bh - 2.0, bw - 2.0, bh))

    def _engine_at(self, pos):
        for e, r in zip(self.engines, self._rects()):
            if r.contains(pos):
                return e, (pos.x() - r.x()) / max(1.0, r.width()), \
                       (pos.y() - r.y()) / max(1.0, r.height())
        return None, 0.0, 0.0

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._pouring = True
            self._pour(e.position())

    def mouseMoveEvent(self, e):
        if self._pouring and (e.buttons() & Qt.LeftButton):
            self._pour(e.position())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._pouring = False

    def leaveEvent(self, e):
        self._pouring = False

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._dirty = True
        if self.paused:
            for eng in self.engines:
                eng.render(eng.last_s)
            self.update()

    def _pour(self, pos):
        e, x, y = self._engine_at(pos)
        if e is None or e.px.size >= MAX_PARTICLES:
            return
        e.pour(x, y, self.brush_sigma)
        self.sandCountChanged.emit(self._total())
        self._dirty = True
        if self.paused:
            e.render(e.last_s)
            self.update()


# =========================================================================== window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chladni Plate Simulator v8")
        self.resize(1250, 890)
        self.setAcceptDrops(True)

        self.layers = []
        self.plate = ChladniPlate()
        self.plate.sandCountChanged.connect(self._update_count)
        self.plate.modeChanged.connect(self._sync_mode_ui)
        self.plate.shapeChanged.connect(self._shape_changed)
        self.plate.recordingChanged.connect(self._recording_changed)

        self.audio = AudioEngine(self)
        self.audio.analysed.connect(self._on_analysed)
        self.audio.layersAnalysed.connect(self._on_layers)
        self.audio.beat.connect(self._on_beat)
        self.audio.positionChanged.connect(self._on_pos)
        self.audio.stateChanged.connect(self._on_state)
        self.audio.finished.connect(self._on_track_finished)

        self.audio_gain = 1.0
        self._dur = 0.0
        self._cur_path = None
        self._last_peak = 0.0
        self._auto_mode = 0
        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self._auto_tick)

        self._cand, self._cand_count = None, 0
        self._rows, self._units = np.zeros((0,)), np.zeros(0)

        root = QWidget()
        lay = QHBoxLayout(root)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(12)
        lay.addWidget(self.plate, 1)
        lay.addWidget(self._build_panel(), 0)
        self.setCentralWidget(root)
        self._build_status()
        self._rebuild_audio_table()

        for t in (QSlider, QPushButton, QComboBox):
            for w in self.findChildren(t):
                w.setFocusPolicy(Qt.NoFocus)

        self._restore_settings()
        self._gen_layers()
        self._apply_layers()
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self._update_fps)
        self.fps_timer.start(700)

    def _build_panel(self):
        plate = self.plate
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setContentsMargins(4, 4, 4, 4)
        col.setSpacing(8)

        def add_slider(parent_lay, label, lo, hi, value, on_change, fmt=str):
            row = QHBoxLayout()
            name = QLabel(label)
            name.setMinimumWidth(92)
            sld = QSlider(Qt.Horizontal)
            sld.setRange(lo, hi)
            sld.setValue(value)
            out = QLabel()
            out.setMinimumWidth(58)
            out.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            def changed(v):
                out.setText(fmt(v))
                on_change(v)

            sld.valueChanged.connect(changed)
            changed(sld.value())
            row.addWidget(name)
            row.addWidget(sld, 1)
            row.addWidget(out)
            parent_lay.addLayout(row)
            return sld

        def button_row(parent_lay, *buttons):
            row = QHBoxLayout()
            for b in buttons:
                row.addWidget(b, 1)
            parent_lay.addLayout(row)

        box = QGroupBox("Shape & manual mode")
        v = QVBoxLayout(box)
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Square plate", "Circular plate"])
        self.shape_combo.currentIndexChanged.connect(
            lambda i: self.plate.set_shape("square" if i == 0 else "circle"))
        v.addWidget(self.shape_combo)
        self.n_slider = add_slider(v, "n", 1, 12, plate.manual[1],
                                   lambda val: plate.set_mode(n=val))
        self.m_slider = add_slider(v, "m", 1, 12, plate.manual[0],
                                   lambda val: plate.set_mode(m=val))
        srow = QHBoxLayout()
        srow.addWidget(QLabel("variant"))
        self.sign_combo = QComboBox()
        self.sign_combo.addItems(["symmetric  (+)", "antisymmetric  (\u2212)"])
        self.sign_combo.currentIndexChanged.connect(
            lambda i: plate.set_mode(sign=1.0 if i == 0 else -1.0))
        srow.addWidget(self.sign_combo, 1)
        v.addLayout(srow)
        self.info = QLabel()
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        col.addWidget(box)

        box = QGroupBox("Frequency layers")
        v = QVBoxLayout(box)
        drow = QHBoxLayout()
        drow.addWidget(QLabel("display"))
        self.display_combo = QComboBox()
        self.display_combo.addItems(["Single plate (overall)",
                                     "Layers \u00b7 superimpose",
                                     "Layers \u00b7 split view (grid)"])
        self.display_combo.currentIndexChanged.connect(lambda _: self._apply_layers())
        drow.addWidget(self.display_combo, 1)
        v.addLayout(drow)

        grow = QHBoxLayout()
        grow.addWidget(QLabel("bands"))
        self.bands_combo = QComboBox()
        self.bands_combo.addItems([f"{n}" for n in BAND_COUNTS])
        self.bands_combo.setCurrentIndex(BAND_COUNTS.index(8))
        self.bands_combo.currentIndexChanged.connect(lambda _: self._bands_changed())
        grow.addWidget(self.bands_combo, 1)
        self.space_combo = QComboBox()
        self.space_combo.addItems(["log", "linear"])
        self.space_combo.currentIndexChanged.connect(lambda _: self._bands_changed())
        grow.addWidget(self.space_combo, 1)
        v.addLayout(grow)

        prow = QHBoxLayout()
        prow.addWidget(QLabel("preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["custom (free drag)",
                                    "octave bands (ISO)",
                                    "\u2153-octave bands (ISO)"])
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        prow.addWidget(self.preset_combo, 1)
        self.snap_chk = QCheckBox("snap \u266a")
        self.snap_chk.toggled.connect(self._snap_toggled)
        prow.addWidget(self.snap_chk)
        v.addLayout(prow)

        self.editor = BandEditor()
        self.editor.setFixedHeight(198)
        self.editor.structureEdited.connect(self._on_structure_edited)
        self.editor.gainEdited.connect(self._on_gains_edited)
        v.addWidget(self.editor)
        ehint = QLabel("drag dividers to re-split \u00b7 double-click a cell mutes it "
                       "\u00b7 lower bars are per-band gain "
                       "(drag \u00b7 Ctrl\u00a0=\u00a0fine \u00d70.1 \u00b7 double-click resets)")
        ehint.setObjectName("hint")
        ehint.setWordWrap(True)
        v.addWidget(ehint)

        jrow = QHBoxLayout()
        save_l = QPushButton("Save layout\u2026")
        save_l.clicked.connect(self._save_layout)
        jrow.addWidget(save_l)
        load_l = QPushButton("Load layout\u2026")
        load_l.clicked.connect(self._load_layout)
        jrow.addWidget(load_l)
        self.flat_btn = QPushButton("Flatten")
        self.flat_btn.setToolTip("Reset every band's gain to 1.0\u00d7")
        self.flat_btn.clicked.connect(self._flatten_gains)
        jrow.addWidget(self.flat_btn)
        self.allon_btn = QPushButton("All on")
        self.allon_btn.setToolTip("Un-mute every band")
        self.allon_btn.clicked.connect(self._all_bands_on)
        jrow.addWidget(self.allon_btn)
        v.addLayout(jrow)
        self.layout_lbl = QLabel("")
        self.layout_lbl.setObjectName("hint")
        v.addWidget(self.layout_lbl)

        self.band_list = QListWidget()
        self.band_list.setFixedHeight(118)
        self.band_list.currentRowChanged.connect(self._on_band_selected)
        v.addWidget(self.band_list)

        orow = QHBoxLayout()
        orow.addWidget(QLabel("mode"))
        self.ov_m = QSpinBox()
        self.ov_m.setRange(1, 12)
        self.ov_m.setPrefix("m ")
        orow.addWidget(self.ov_m)
        self.ov_n = QSpinBox()
        self.ov_n.setRange(1, 12)
        self.ov_n.setPrefix("n ")
        orow.addWidget(self.ov_n)
        self.ov_sign = QComboBox()
        self.ov_sign.addItems(["sym +", "antisym \u2212"])
        self.ov_sign.setFixedWidth(74)
        orow.addWidget(self.ov_sign)
        self.ov_set = QPushButton("Set")
        self.ov_set.clicked.connect(self._override_set)
        orow.addWidget(self.ov_set)
        self.ov_auto = QPushButton("Auto")
        self.ov_auto.clicked.connect(self._override_auto)
        orow.addWidget(self.ov_auto)
        v.addLayout(orow)
        col.addWidget(box)

        box = QGroupBox("Music  \u2014  play songs on the plate(s)")
        v = QVBoxLayout(box)
        arow = QHBoxLayout()
        add_btn = QPushButton("Add songs\u2026")
        add_btn.clicked.connect(self._add_songs)
        arow.addWidget(add_btn)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._remove_song)
        arow.addWidget(rm_btn)
        clr_btn = QPushButton("Clear")
        clr_btn.clicked.connect(self._clear_playlist)
        arow.addWidget(clr_btn)
        v.addLayout(arow)

        self.playlist = QListWidget()
        self.playlist.setFixedHeight(72)
        self.playlist.itemDoubleClicked.connect(
            lambda it: self._play_index(self.playlist.row(it)))
        v.addWidget(self.playlist)

        prow2 = QHBoxLayout()
        self.prev_btn = QPushButton("\u23ee")
        self.prev_btn.setFixedWidth(36)
        self.prev_btn.clicked.connect(self._prev_track)
        self.play_btn = QPushButton("\u25b6 Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.stop_btn = QPushButton("\u23f9")
        self.stop_btn.setFixedWidth(34)
        self.stop_btn.clicked.connect(self.audio.stop)
        self.next_btn = QPushButton("\u23ed")
        self.next_btn.setFixedWidth(36)
        self.next_btn.clicked.connect(self._next_track)
        prow2.addWidget(self.prev_btn)
        prow2.addWidget(self.play_btn)
        prow2.addWidget(self.stop_btn)
        prow2.addWidget(self.next_btn)
        v.addLayout(prow2)

        seek_row = QHBoxLayout()
        self.pos_slider = QSlider(Qt.Horizontal)
        self.pos_slider.setRange(0, 0)
        self.pos_slider.setEnabled(False)
        self.pos_slider.valueChanged.connect(self._pos_scrub)
        seek_row.addWidget(self.pos_slider, 1)
        self.time_lbl = QLabel("0:00 / 0:00")
        self.time_lbl.setMinimumWidth(78)
        self.time_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        seek_row.addWidget(self.time_lbl)
        v.addLayout(seek_row)

        self.track_lbl = QLabel("no song loaded")
        self.track_lbl.setObjectName("hint")
        v.addWidget(self.track_lbl)

        add_slider(v, "volume", 0, 100, 80, lambda val: self.audio.set_volume(val / 100.0))
        lrow = QHBoxLayout()
        self.loop_chk = QCheckBox("loop track")
        lrow.addWidget(self.loop_chk)
        self.autoadv_chk = QCheckBox("auto-advance")
        self.autoadv_chk.setChecked(True)
        lrow.addWidget(self.autoadv_chk, 1)
        v.addLayout(lrow)
        self.tone_btn = QPushButton("Play test tone")
        self.tone_btn.clicked.connect(self.audio.toggle_tone)
        v.addWidget(self.tone_btn)
        self.tone_slider = add_slider(
            v, "tone freq", 0, 1000, 521,
            lambda val: self.audio.set_tone_freq(40.0 * 100.0 ** (val / 1000.0)),
            lambda val: f"{40.0 * 100.0 ** (val / 1000.0):.0f} Hz")
        col.addWidget(box)

        box = QGroupBox("Audio \u2192 plate")
        v = QVBoxLayout(box)
        self.react_chk = QCheckBox("audio drives plate(s)")
        self.react_chk.toggled.connect(self._react_toggled)
        v.addWidget(self.react_chk)
        add_slider(v, "audio gain", 10, 300, 100,
                   lambda val: setattr(self, "audio_gain", val / 100.0),
                   lambda val: f"{val / 100.0:.1f}")
        self.beat_chk = QCheckBox("beats kick the sand")
        self.beat_chk.setChecked(True)
        v.addWidget(self.beat_chk)
        self.spec_chk = QCheckBox("spectrum overlay")
        self.spec_chk.toggled.connect(plate.set_show_spectrum)
        v.addWidget(self.spec_chk)
        self.audio_map_lbl = QLabel("peak \u2014 Hz")
        self.audio_map_lbl.setObjectName("hint")
        v.addWidget(self.audio_map_lbl)
        col.addWidget(box)

        box = QGroupBox("Sand")
        v = QVBoxLayout(box)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("grains / plate"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(0, MAX_PARTICLES)
        self.count_spin.setSingleStep(500)
        self.count_spin.setValue(4000)
        self.count_spin.valueChanged.connect(plate.set_particle_count)
        crow.addWidget(self.count_spin, 1)
        v.addLayout(crow)
        add_slider(v, "brush size", 1, 30, 8,
                   lambda val: setattr(plate, "brush_sigma", 0.004 + val * 0.004))
        add_slider(v, "agitation", 0, 100, 35,
                   lambda val: setattr(plate, "speed", val * 0.03),
                   lambda val: f"{val * 0.03:.2f}")
        add_slider(v, "jitter", 0, 100, 20,
                   lambda val: setattr(plate, "jitter", val * 4e-5),
                   lambda val: f"{val * 4e-5:.4f}")
        self.momentum_chk = QCheckBox("grain inertia (momentum)")
        self.momentum_chk.setChecked(plate.momentum)
        self.momentum_chk.toggled.connect(lambda on: setattr(plate, "momentum", on))
        v.addWidget(self.momentum_chk)
        add_slider(v, "drag", 0, 90, 65,
                   lambda val: setattr(plate, "drag", val / 100.0),
                   lambda val: f"{val / 100.0:.2f}")
        sc_btn = QPushButton("Scatter")
        cl_btn = QPushButton("Clear")
        tp_btn = QPushButton("Tap plate")
        sc_btn.clicked.connect(lambda: plate.reset_sand(self.count_spin.value()))
        cl_btn.clicked.connect(lambda: plate.reset_sand(0))
        tp_btn.clicked.connect(plate.tap)
        button_row(v, sc_btn, cl_btn, tp_btn)
        col.addWidget(box)

        box = QGroupBox("Look")
        v = QVBoxLayout(box)
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("colormap"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(list(COLORMAPS))
        self.cmap_combo.currentIndexChanged.connect(plate.set_colormap)
        lrow.addWidget(self.cmap_combo, 1)
        v.addLayout(lrow)
        prow3 = QHBoxLayout()
        prow3.addWidget(QLabel("grain size"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["1 px", "2 px", "3 px"])
        self.size_combo.setCurrentIndex(1)
        self.size_combo.currentIndexChanged.connect(lambda i: plate.set_particle_size(i + 1))
        prow3.addWidget(self.size_combo, 1)
        self.sand_btn = QPushButton("sand color")
        self.sand_btn.clicked.connect(self._pick_sand_color)
        prow3.addWidget(self.sand_btn, 1)
        v.addLayout(prow3)
        self.trail_chk = QCheckBox("motion trails (long exposure)")
        self.trail_chk.toggled.connect(plate.set_trails)
        v.addWidget(self.trail_chk)
        self.trail_slider = add_slider(v, "trail decay", 60, 97, 88,
                                       lambda val: setattr(plate, "trail_decay", val / 100.0),
                                       lambda val: f"0.{val}")
        self.trail_slider.setEnabled(False)
        self.trail_chk.toggled.connect(self.trail_slider.setEnabled)
        col.addWidget(box)

        box = QGroupBox("Vibration & auto-advance")
        v = QVBoxLayout(box)
        vib = QCheckBox("animate plate oscillation")
        vib.setChecked(plate.vibrate)
        vib.toggled.connect(lambda on: setattr(plate, "vibrate", on))
        v.addWidget(vib)
        add_slider(v, "amplitude", 20, 100, 100,
                   lambda val: setattr(plate, "amp", val / 100.0),
                   lambda val: f"{val / 100.0:.2f}")
        add_slider(v, "frequency", 5, 200, 60,
                   lambda val: setattr(plate, "vib_freq", val / 100.0),
                   lambda val: f"{val / 100.0:.2f} Hz")
        arow = QHBoxLayout()
        self.auto_combo = QComboBox()
        self.auto_combo.addItems(["auto-advance: off", "random modes", "frequency sweep"])
        self.auto_combo.currentIndexChanged.connect(self._auto_changed)
        arow.addWidget(self.auto_combo, 1)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 30)
        self.interval_spin.setValue(6)
        self.interval_spin.setSuffix(" s")
        self.interval_spin.valueChanged.connect(self._interval_changed)
        arow.addWidget(self.interval_spin)
        v.addLayout(arow)
        col.addWidget(box)

        box = QGroupBox("Actions")
        v = QVBoxLayout(box)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._toggle_pause)
        rnd = QPushButton("Random mode")
        rnd.clicked.connect(plate.randomize_mode)
        abt = QPushButton("About")
        abt.clicked.connect(self._about)
        button_row(v, self.pause_btn, rnd, abt)
        save = QPushButton("Save PNG\u2026")
        save.clicked.connect(self._save_png)
        self.rec_btn = QPushButton("Record GIF")
        self.rec_btn.clicked.connect(self._toggle_record)
        button_row(v, save, self.rec_btn)
        hint = QLabel("Drag on a plate to pour sand \u00b7 drop songs on the window.\n"
                      "m/n sliders set the fallback pattern; audio overrides per band "
                      "(unless a band is hand-locked with \u270e).\n"
                      "Keys: \u2190\u2192 n \u00b7 \u2191\u2193 m \u00b7 R random \u00b7 Space pause \u00b7 "
                      "T tap \u00b7 S save \u00b7 G gif \u00b7 O songs \u00b7 A play \u00b7 N/B track \u00b7 "
                      "Y tone \u00b7 1/2 shape")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        col.addWidget(box)

        col.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(380)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        return scroll

    def _gen_layers(self):
        count = BAND_COUNTS[self.bands_combo.currentIndex()]
        log = self.space_combo.currentIndex() == 0
        edges = (np.geomspace(FMIN, FMAX, count + 1) if log
                 else np.linspace(FMIN, FMAX, count + 1))
        self._build_layers_from_edges(edges)

    def _build_layers_from_edges(self, edges, centers=None):
        count = len(edges) - 1
        self.layers = []
        for i in range(count):
            lo, hi = float(edges[i]), float(edges[i + 1])
            name = (center_name(centers[i])
                    if centers and i < len(centers) else band_name(lo, hi))
            self.layers.append(Layer(
                name=name, lo=lo, hi=hi, color=band_color(i),
                m=min(12, 1 + int(round(11 * i / max(1, count - 1)))),
                n=1 + (i * 3) % 5, sign=1.0, gain=1.0))

    def _apply_layers(self):
        d = ("single", "super", "split")[self.display_combo.currentIndex()]
        layered = d != "single" and bool(self.layers)
        self.editor.setEnabled(layered)
        self.editor.set_layers(self.layers if layered else [])
        self._rebuild_band_list()
        if not layered:
            self.audio.set_bands([])
            self.plate.set_display("single", [])
            self._update_layer_info()
            return
        self.audio.set_bands([(L.lo, L.hi) for L in self.layers])
        self.plate.set_display(d, self.layers)
        self._push_layers_fallback()

    def _bands_changed(self):
        self._gen_layers()
        self._apply_layers()

    def _preset_changed(self, idx):
        if idx == 0:
            self.bands_combo.setEnabled(True)
            self.space_combo.setEnabled(True)
            return
        if idx == 1:
            edges = preset_edges(OCTAVE_CENTERS, 1)
            centers = OCTAVE_CENTERS
        else:
            edges = preset_edges(THIRD_OCTAVE_CENTERS, 3)
            centers = THIRD_OCTAVE_CENTERS
        self.bands_combo.setEnabled(False)
        self.space_combo.setEnabled(False)
        self._build_layers_from_edges(edges, centers)
        self._apply_layers()

    def _snap_toggled(self, on):
        self.editor.snap = bool(on)
        self.editor.update()

    def _on_structure_edited(self, kind):
        if kind == "div" and self.preset_combo.currentIndex() != 0:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(0)
            self.preset_combo.blockSignals(False)
            self.bands_combo.setEnabled(True)
            self.space_combo.setEnabled(True)
        if self.plate.display == "single" or not self.layers:
            return
        self.audio.set_bands([(L.lo, L.hi) for L in self.layers])
        en = [L for L in self.layers if L.enabled] or self.layers[:1]
        if self.plate.display == "split" and len(self.plate.engines) != len(en):
            self.plate.set_display("split", self.layers)
        else:
            self.plate.refresh_layer_metadata()
        if not (self.react_chk.isChecked() and self.audio.state == "playing"):
            self._push_layers_fallback()
        self._update_layer_info()

    def _on_gains_edited(self):
        if self.plate.display == "single" or not self.layers:
            return
        if not (self.react_chk.isChecked() and self.audio.state == "playing"):
            self._push_layers_fallback()
        self._update_layer_info()

    def _flatten_gains(self):
        for L in self.layers:
            L.gain = 1.0
        self.editor.update()
        self._on_gains_edited()

    def _all_bands_on(self):
        for L in self.layers:
            L.enabled = True
        self.editor.update()
        self._on_structure_edited("mute")

    def _push_layers_fallback(self):
        p = self.plate
        if p.display == "single" or not p.layers:
            return
        if p.display == "super":
            modes, weights = [], []
            for i, L in enumerate(p.layers):
                if L.override:
                    modes.append((L.m, L.n, L.sign))
                else:
                    modes.append((min(p.manual[0] + i, 12), max(1, p.manual[1]),
                                  p.manual[2]))
                weights.append(0.55 * L.gain if L.enabled else 0.0)
            p.set_super(modes, weights)
        else:
            items = []
            for i, L in enumerate(p.layers):
                if not L.enabled:
                    continue
                md = ((L.m, L.n, L.sign) if L.override
                      else (min(p.manual[0] + i, 12), max(1, p.manual[1]),
                            p.manual[2]))
                items.append((md, 0.55 * L.gain))
            if not items:
                items = [((p.manual[0], p.manual[1], p.manual[2]), 0.55)]
            p.set_split_modes(items)
        self._update_layer_info()

    def _rebuild_audio_table(self):
        p = self.plate
        if p.shape == "square":
            rows = [(m * m + n * n, m, n, s) for m in range(1, 13)
                    for n in range(1, 13) for s in (1.0, -1.0) if not (s < 0 and m == n)]
        else:
            rows = [(bessel_root(m, n) ** 2, m, n, 1.0)
                    for m in range(0, 9) for n in range(1, 9)]
        rows.sort()
        self._rows = rows
        self._units = np.array([r[0] for r in rows])

    def _map_freq_to_mode(self, f, lo=None, hi=None):
        if self._units.size == 0:
            return None
        a = np.log(max(lo, 20.0)) if lo else np.log(150.0)
        b = np.log(min(hi, 8000.0)) if hi else np.log(3000.0)
        if b <= a:
            b = a + 0.5
        t = min(max((np.log(max(f, 1.0)) - a) / (b - a), 0.0), 1.0)
        target = self._units[0] + t * (self._units[-1] - self._units[0])
        i = int(np.argmin(np.abs(self._units - target)))
        u0 = self._units[i]
        best, bd = None, 1e18
        p = self.plate
        for r in self._rows:
            if abs(r[0] - u0) <= 1e-6 * max(1.0, abs(u0)):
                d = abs(r[1] - p.manual[0]) + abs(r[2] - p.manual[1])
                if p.shape == "square" and r[3] != p.manual[2]:
                    d += 0.5
                if d < bd:
                    bd, best = d, r
        return best

    def _rebuild_band_list(self):
        self.band_list.blockSignals(True)
        self.band_list.clear()
        for L in self.layers:
            it = QListWidgetItem()
            it.setForeground(QColor(*L.color))
            self.band_list.addItem(it)
        self.band_list.blockSignals(False)
        if self.layers:
            self.band_list.setCurrentRow(0)
        self._sync_override_controls()

    def _cur_band(self):
        r = self.band_list.currentRow()
        if 0 <= r < len(self.layers):
            return self.layers[r]
        return None

    def _sync_override_controls(self):
        L = self._cur_band()
        ok = self.plate.display != "single" and L is not None
        for w in (self.ov_m, self.ov_n, self.ov_sign, self.ov_set, self.ov_auto):
            w.setEnabled(ok)
        if L is not None:
            for w in (self.ov_m, self.ov_n, self.ov_sign):
                w.blockSignals(True)
            self.ov_m.setValue(int(np.clip(L.m, self.ov_m.minimum(),
                                           self.ov_m.maximum())))
            self.ov_n.setValue(int(np.clip(L.n, 1, 12)))
            self.ov_sign.setCurrentIndex(0 if L.sign > 0 else 1)
            for w in (self.ov_m, self.ov_n, self.ov_sign):
                w.blockSignals(False)

    def _on_band_selected(self, _row):
        self._sync_override_controls()

    def _override_set(self):
        L = self._cur_band()
        if L is None:
            return
        L.m = int(self.ov_m.value())
        L.n = int(self.ov_n.value())
        L.sign = 1.0 if self.ov_sign.currentIndex() == 0 else -1.0
        L.override = True
        self._update_layer_info()
        if not (self.react_chk.isChecked() and self.audio.state == "playing"):
            self._push_layers_fallback()

    def _override_auto(self):
        L = self._cur_band()
        if L is None:
            return
        L.override = False
        if L.peak > 20:
            r = self._map_freq_to_mode(L.peak, L.lo, L.hi)
            if r:
                L.m, L.n, L.sign = int(r[1]), int(r[2]), float(r[3])
        self._update_layer_info()
        if not (self.react_chk.isChecked() and self.audio.state == "playing"):
            self._push_layers_fallback()

    def _update_one_layer(self, L, i, data, reactive):
        if not reactive or i >= len(data):
            if not reactive and not L.override:
                L.m = min(self.plate.manual[0] + i, 12)
                L.n = max(1, self.plate.manual[1])
                L.sign = self.plate.manual[2]
            L.energy = 0.5 if not reactive else 0.0
            return
        e, pk = data[i]
        L.env = max(e, L.env * 0.90)
        L.energy = min(1.0, (e / (L.env + 1e-9))) ** 0.6
        if pk > 0:
            L.peak = pk
        if L.override:
            return
        if L.energy > 0.05 and L.peak > 20:
            r = self._map_freq_to_mode(L.peak, L.lo, L.hi)
            if r:
                same = (r[1] == L.m and r[2] == L.n
                        and (self.plate.shape != "square" or r[3] == L.sign))
                if same:
                    L.cand, L.cand_count = None, 0
                elif L.cand == r:
                    L.cand_count += 1
                    if L.cand_count >= 3:
                        L.m, L.n, L.sign = int(r[1]), int(r[2]), float(r[3])
                        L.cand, L.cand_count = None, 0
                else:
                    L.cand, L.cand_count = r, 1

    def _on_layers(self, data):
        p = self.plate
        if p.display == "single" or not p.layers:
            return
        reactive = self.react_chk.isChecked()
        if p.display == "super":
            modes, weights = [], []
            for i, L in enumerate(p.layers):
                self._update_one_layer(L, i, data, reactive)
                modes.append((L.m, L.n, L.sign))
                weights.append(L.gain * (0.06 + 0.94 * L.energy)
                               if L.enabled else 0.0)
            p.set_super(modes, weights)
        else:
            idxs = [i for i, L in enumerate(p.layers) if L.enabled] or [0]
            items = []
            for li in idxs:
                L = p.layers[li]
                self._update_one_layer(L, li, data, reactive)
                wt = L.gain * (0.06 + 0.94 * L.energy) if reactive else 0.55 * L.gain
                items.append(((L.m, L.n, L.sign), wt))
            p.set_split_modes(items)
        self._update_layer_info()

    def _band_item_text(self, L):
        if not L.enabled:
            return f"\u25aa {L.name}: off"
        sq = self.plate.shape == "square"
        var = (("+" if L.sign > 0 else "\u2212") if sq
               else ("cos" if L.sign > 0 else "sin"))
        mark = "\u270e" if L.override else "\u00b7"
        return (f"\u25b8 {L.name} \u00b7 {L.peak:.0f} Hz \u00b7 E {L.energy:.2f} "
                f"\u00b7 {L.gain:.1f}\u00d7 \u00b7 {mark}({L.m},{L.n}){var}")

    def _update_layer_info(self):
        if not hasattr(self, "band_list"):
            return
        if self.plate.display == "single" or not self.layers:
            if self.band_list.count():
                self.band_list.clear()
            self._sync_override_controls()
            if hasattr(self, "editor"):
                self.editor.update()
            return
        for i, L in enumerate(self.layers):
            if i < self.band_list.count():
                self.band_list.item(i).setText(self._band_item_text(L))
        if hasattr(self, "editor"):
            self.editor.update()

    def _save_layout(self):
        if not self.layers:
            QMessageBox.information(self, "No bands", "There are no bands to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save band layout",
                                              "chladni_bands.json",
                                              "JSON (*.json);;All files (*)")
        if not path:
            return
        data = {
            "app": "chladni-plate",
            "version": 2,
            "fmin": FMIN,
            "fmax": FMAX,
            "snap_notes": self.snap_chk.isChecked(),
            "bands": [{"lo": round(L.lo, 3), "hi": round(L.hi, 3),
                       "gain": round(L.gain, 3), "enabled": bool(L.enabled),
                       "override": bool(L.override),
                       "m": int(L.m), "n": int(L.n), "sign": float(L.sign)}
                      for L in self.layers],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.layout_lbl.setText(os.path.basename(path))
        except Exception as ex:
            QMessageBox.warning(self, "Save failed", str(ex))

    def _load_layout(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load band layout", "",
                                              "JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            bands = data["bands"]
            if not isinstance(bands, list) or not bands:
                raise ValueError("no bands in file")
            layers = []
            for b in bands:
                lo = float(np.clip(float(b["lo"]), FMIN, FMAX))
                hi = float(np.clip(float(b["hi"]), FMIN, FMAX))
                if hi <= lo:
                    continue
                layers.append(Layer(
                    name=band_name(lo, hi), lo=lo, hi=hi,
                    color=band_color(len(layers)),
                    gain=float(np.clip(float(b.get("gain", 1.0)), 0.0, GAIN_MAX)),
                    enabled=bool(b.get("enabled", True)),
                    override=bool(b.get("override", False)),
                    m=int(np.clip(int(b.get("m", 2)), 0, 12)),
                    n=int(np.clip(int(b.get("n", 2)), 1, 12)),
                    sign=1.0 if float(b.get("sign", 1.0)) >= 0 else -1.0))
            if not layers:
                raise ValueError("no valid bands")
            prev = layers[0].lo
            for L in layers[1:]:
                L.lo = max(L.lo, prev * 1.02)
                prev = L.hi
            self.layers = layers
        except Exception as ex:
            QMessageBox.warning(self, "Load failed", f"{path}\n\n{ex}")
            return
        snap = bool(data.get("snap_notes", False))
        self.snap_chk.blockSignals(True)
        self.snap_chk.setChecked(snap)
        self.snap_chk.blockSignals(False)
        self.editor.snap = snap
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)
        self.bands_combo.setEnabled(True)
        self.space_combo.setEnabled(True)
        self.layout_lbl.setText(os.path.basename(path))
        self._apply_layers()

    def _add_to_playlist(self, path):
        for i in range(self.playlist.count()):
            if self.playlist.item(i).data(Qt.UserRole) == path:
                return i
        it = QListWidgetItem(os.path.basename(path))
        it.setData(Qt.UserRole, path)
        it.setToolTip(path)
        self.playlist.addItem(it)
        return self.playlist.count() - 1

    def _add_songs(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add songs", "",
            "Audio (*.wav *.mp3 *.ogg *.flac *.aif *.aiff);;All files (*)")
        first = None
        for p in paths:
            r = self._add_to_playlist(p)
            if first is None:
                first = r
        if first is not None and self.audio.state != "playing":
            self._play_index(first)

    def _remove_song(self):
        for it in self.playlist.selectedItems():
            if it.data(Qt.UserRole) == self._cur_path:
                self.audio.stop()
                self._cur_path = None
            self.playlist.takeItem(self.playlist.row(it))

    def _clear_playlist(self):
        self.audio.stop()
        self.playlist.clear()
        self._cur_path = None
        self.track_lbl.setText("no song loaded")

    def _mark_playing(self, path):
        for i in range(self.playlist.count()):
            it = self.playlist.item(i)
            base = os.path.basename(it.data(Qt.UserRole))
            it.setText(("\u25b6 " if it.data(Qt.UserRole) == path else "") + base)

    def _load_track(self, path):
        if not HAVE_QTMULTIMEDIA:
            QMessageBox.warning(self, "No audio backend",
                                "PySide6 QtMultimedia is not available.")
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            data, sr = decode_audio(path)
        except Exception as ex:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Decode error",
                                f"Could not read:\n{path}\n\n{ex}")
            return False
        QApplication.restoreOverrideCursor()
        n = self.audio.load(data, sr)
        self._dur = n / SR
        self._cur_path = path
        self.track_lbl.setText(os.path.basename(path))
        self.pos_slider.blockSignals(True)
        self.pos_slider.setRange(0, n)
        self.pos_slider.setValue(0)
        self.pos_slider.blockSignals(False)
        self.pos_slider.setEnabled(True)
        self.time_lbl.setText(f"0:00 / {fmt_time(self._dur)}")
        self._mark_playing(path)
        return True

    def _play_index(self, i):
        if not (0 <= i < self.playlist.count()):
            return
        self.playlist.setCurrentRow(i)
        path = self.playlist.item(i).data(Qt.UserRole)
        if self._load_track(path):
            self.audio.play()

    def _next_track(self):
        n = self.playlist.count()
        if n == 0:
            return
        self._play_index((self.playlist.currentRow() + 1) % n)

    def _prev_track(self):
        if self.audio.pos / SR > 3.0:
            self.audio.seek(0)
            return
        n = self.playlist.count()
        if n == 0:
            return
        self._play_index((self.playlist.currentRow() - 1) % n)

    def _on_track_finished(self):
        if self.autoadv_chk.isChecked() and self.playlist.count():
            self._next_track()

    def _on_analysed(self, rms, peak, strong):
        self.level_bar.setValue(int(min(100, rms * 300)))
        if peak > 0:
            self._last_peak = peak
        self.freq_lbl.setText(f"{peak:.0f} Hz" if peak > 0 else "\u2014")
        self.plate.set_spectrum(self.audio.spec.copy())

        if self.plate.display == "single" or not self.plate.layers:
            if self.react_chk.isChecked():
                self.plate.set_audio_drive(min(1.6, 0.06 + rms * self.audio_gain * 4.0))
                if not strong or peak < 30:
                    self._cand, self._cand_count = None, 0
                else:
                    r = self._map_freq_to_mode(peak)
                    p = self.plate
                    m0, n0, s0 = p.layer_mode()
                    cur = (m0, n0, s0 if p.shape == "square" else 1.0)
                    if r is None or (r[1], r[2],
                                     r[3] if p.shape == "square" else 1.0) == cur:
                        self._cand, self._cand_count = None, 0
                    elif self._cand == r:
                        self._cand_count += 1
                        if self._cand_count >= 3:
                            p.set_mode(r[1], r[2],
                                       r[3] if p.shape == "square" else None)
                            self._cand, self._cand_count = None, 0
                    else:
                        self._cand, self._cand_count = r, 1
            self.audio_map_lbl.setText(f"peak {self._last_peak:.0f} Hz")

    def _on_beat(self, strength):
        if self.beat_chk.isChecked():
            self.plate.beat_pulse(strength)

    def _react_toggled(self, on):
        self.plate.audio_reactive = on
        if not on:
            self.plate.set_audio_drive(1.0)
            self._push_layers_fallback()

    def _on_state(self, state):
        playing = state == "playing"
        self.play_btn.setText("\u23f8 Pause" if playing else "\u25b6 Play")
        self.tone_btn.setText("Stop tone" if (playing and self.audio.source == "tone")
                              else "Play test tone")
        if not playing:
            self._push_layers_fallback()

    def _pos_scrub(self, v):
        if self.pos_slider.isSliderDown():
            self.audio.seek(v)

    def _on_pos(self, t):
        if self.audio.source != "file":
            return
        if not self.pos_slider.isSliderDown():
            self.pos_slider.blockSignals(True)
            self.pos_slider.setValue(int(t * SR))
            self.pos_slider.blockSignals(False)
        self.time_lbl.setText(f"{fmt_time(t)} / {fmt_time(self._dur)}")

    def _toggle_play(self):
        if self.audio.source is None and self.playlist.count():
            self._play_index(max(0, self.playlist.currentRow()))
            return
        if not self.audio.toggle():
            QMessageBox.information(
                self, "No audio",
                "Add songs to the playlist (O) or start the tone generator (Y).")

    def _build_status(self):
        sb = self.statusBar()
        self.rec_lbl = QLabel("")
        self.mode_lbl = QLabel("")
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedWidth(120)
        self.level_bar.setFixedHeight(12)
        self.freq_lbl = QLabel("\u2014")
        self.count_lbl = QLabel("")
        self.fps_lbl = QLabel("")
        for w in (self.rec_lbl, self.mode_lbl, self.level_bar,
                  self.freq_lbl, self.count_lbl, self.fps_lbl):
            sb.addPermanentWidget(w)

    def _shape_changed(self, shape):
        circle = shape == "circle"
        (mlo, mhi), (nlo, nhi) = PlateEngine._ranges(shape)
        self.m_slider.setRange(mlo, mhi)
        self.n_slider.setRange(nlo, nhi)
        self.sign_combo.setItemText(0, "cos(m\u03b8)" if circle else "symmetric  (+)")
        self.sign_combo.setItemText(1, "sin(m\u03b8)" if circle else "antisymmetric  (\u2212)")
        self.ov_m.setRange(mlo, mhi)
        self.ov_n.setRange(nlo, nhi)
        self.ov_sign.setItemText(0, "cos" if circle else "sym +")
        self.ov_sign.setItemText(1, "sin" if circle else "antisym \u2212")
        self._rebuild_audio_table()
        self._push_layers_fallback()
        self._sync_override_controls()

    def _sync_mode_ui(self, m, n, sign):
        for w in (self.m_slider, self.n_slider, self.sign_combo):
            w.blockSignals(True)
        self.m_slider.setValue(m)
        self.n_slider.setValue(n)
        self.sign_combo.setCurrentIndex(0 if sign > 0 else 1)
        for w in (self.m_slider, self.n_slider, self.sign_combo):
            w.blockSignals(False)
        self._refresh_info()

    def _refresh_info(self):
        if not hasattr(self, "info"):
            return
        p = self.plate
        e = p.engines[0] if p.engines else None
        if e is None:
            return
        if p.shape == "square":
            rel = e.m * e.m + e.n * e.n
            note = "\n\u26a0 degenerate: flat plate" if (e.sign < 0 and e.m == e.n) else ""
            self.info.setText(f"manual mode ({e.m}, {e.n})  "
                              f"{'+' if e.sign > 0 else '\u2212'}\n"
                              f"f \u221d m\u00b2 + n\u00b2 = {rel}{note}")
            short = f"({e.m},{e.n}) {'+' if e.sign > 0 else '\u2212'}"
        else:
            k = bessel_root(e.m, e.n)
            self.info.setText(
                f"manual mode ({e.m}, {e.n})  "
                f"{'cos m\u03b8' if e.sign > 0 else 'sin m\u03b8'}\n"
                f"root \u03bb = {k:.3f}    f \u221d \u03bb\u00b2 = {k ** 2:.1f}")
            short = f"({e.m},{e.n}) {'cos' if e.sign > 0 else 'sin'}"
        self.mode_lbl.setText(f"mode {short}  ·  {p.shape}")

    def _update_count(self, n):
        self.count_lbl.setText(f"{n:,} grains")

    def _update_fps(self):
        self.fps_lbl.setText(f"{self.plate.take_fps() * 10 // 7} fps")
        if self.plate._recording:
            self.rec_lbl.setText(
                f"<span style='color:#ff5555'>\u25cf REC {len(self.plate._rec_frames)}</span>")

    def _toggle_pause(self, checked):
        self.plate.paused = checked
        self.plate.mark_dirty()
        self.pause_btn.setText("Resume" if checked else "Pause")

    def _auto_changed(self, idx):
        self._auto_mode = idx
        self.auto_timer.stop()
        if idx:
            self.auto_timer.start(self.interval_spin.value() * 1000)

    def _interval_changed(self, val):
        if self._auto_mode:
            self.auto_timer.start(val * 1000)

    def _auto_tick(self):
        if self._auto_mode == 1:
            self.plate.randomize_mode()
        elif self._auto_mode == 2:
            self.plate.next_sweep()

    def _pick_sand_color(self):
        c = QColorDialog.getColor(QColor(*self.plate._sand_rgb), self, "Sand color")
        if c.isValid():
            self.plate.set_sand_color(c.red(), c.green(), c.blue())
            self.sand_btn.setStyleSheet(
                f"background: rgb({c.red()},{c.green()},{c.blue()});")

    def _save_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save snapshot",
                                              "chladni.png", "PNG images (*.png)")
        if path:
            self.plate._compose().save(path)

    def _toggle_record(self):
        p = self.plate
        if p._recording:
            p.stop_recording()
        else:
            if not HAVE_IMAGEIO:
                QMessageBox.warning(self, "imageio missing",
                                    "GIF export needs imageio + pillow:\n\n"
                                    "pip install imageio pillow")
                return
            p.start_recording()

    def _recording_changed(self, on):
        if on:
            self.rec_btn.setText("Stop GIF \u23f9")
            self.rec_lbl.setText("<span style='color:#ff5555'>\u25cf REC</span>")
        else:
            self.rec_btn.setText("Record GIF")
            self.rec_lbl.setText("")
            frames = self.plate.pop_recording()
            if frames:
                path, _ = QFileDialog.getSaveFileName(self, "Save GIF",
                                                      "chladni.gif", "GIF (*.gif)")
                if path:
                    try:
                        imageio.mimsave(path, frames, fps=30)
                    except TypeError:
                        imageio.mimsave(path, frames, duration=1.0 / 30.0, loop=0)

    def _about(self):
        decoders = [n for n, ok in (("miniaudio", _miniaudio is not None),
                                    ("soundfile", _sf is not None)) if ok]
        QMessageBox.about(
            self, "About",
            "<b>Chladni Plate Simulator v8</b><br><br>"
            "Complete edition: square & circular plates, song playlist with "
            "beat detection, EQ-style granular frequency layers (ISO presets, "
            "note snapping, per-band gains with Ctrl-fine drag, mode "
            "overrides), JSON layouts, GIF/PNG export.<br><br>"
            f"Decoders: {', '.join(decoders) or 'stdlib wave only'}. "
            "For MP3/OGG/FLAC: <code>pip install miniaudio</code>")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.toLocalFile().lower().endswith(AUDIO_EXTS):
                    e.acceptProposedAction()
                    return
        super().dragEnterEvent(e)

    def dropEvent(self, e):
        added, first = False, None
        for u in e.mimeData().urls():
            f = u.toLocalFile()
            if f.lower().endswith(AUDIO_EXTS):
                r = self._add_to_playlist(f)
                added = True
                if first is None:
                    first = r
        if added:
            e.acceptProposedAction()
            if self.audio.state != "playing":
                self._play_index(first)
        else:
            super().dropEvent(e)

    def keyPressEvent(self, e):
        k, p = e.key(), self.plate
        if k == Qt.Key_Space:
            self.pause_btn.toggle()
        elif k == Qt.Key_R:
            p.randomize_mode()
        elif k == Qt.Key_T:
            p.tap()
        elif k == Qt.Key_S:
            self._save_png()
        elif k == Qt.Key_G:
            self._toggle_record()
        elif k == Qt.Key_O:
            self._add_songs()
        elif k == Qt.Key_A:
            self._toggle_play()
        elif k == Qt.Key_Y:
            self.audio.toggle_tone()
        elif k == Qt.Key_N:
            self._next_track()
        elif k == Qt.Key_B:
            self._prev_track()
        elif k == Qt.Key_1:
            self.shape_combo.setCurrentIndex(0)
        elif k == Qt.Key_2:
            self.shape_combo.setCurrentIndex(1)
        elif k == Qt.Key_Left:
            p.set_mode(n=p.manual[1] - 1)
        elif k == Qt.Key_Right:
            p.set_mode(n=p.manual[1] + 1)
        elif k == Qt.Key_Up:
            p.set_mode(m=p.manual[0] + 1)
        elif k == Qt.Key_Down:
            p.set_mode(m=p.manual[0] - 1)
        else:
            super().keyPressEvent(e)

    def _restore_settings(self):
        s = QSettings("chladni", "chladni-plate")
        geom = s.value("geometry")
        if geom is not None:
            self.restoreGeometry(geom)
        self.shape_combo.setCurrentIndex(
            0 if s.value("shape", "square") == "square" else 1)
        m, n_, sg = (s.value("m", 3, type=int), s.value("n", 2, type=int),
                     s.value("sign", -1.0, type=float))
        self.plate.manual = (m, n_, sg)
        self.plate.engines[0].set_mode(m, n_, sg)
        self.count_spin.setValue(s.value("grains", 4000, type=int))
        self.cmap_combo.setCurrentIndex(s.value("cmap", 0, type=int))
        self.loop_chk.setChecked(s.value("loop", False, type=bool))
        self.autoadv_chk.setChecked(s.value("autoadv", True, type=bool))
        self.react_chk.setChecked(s.value("reactive", False, type=bool))
        self.beat_chk.setChecked(s.value("beats", True, type=bool))
        self.spec_chk.setChecked(s.value("spectrum", False, type=bool))
        self.snap_chk.setChecked(s.value("snap", False, type=bool))
        self.audio_gain = s.value("again", 100, type=int) / 100.0
        for w in (self.display_combo, self.bands_combo, self.space_combo):
            w.blockSignals(True)
        self.display_combo.setCurrentIndex(s.value("display", 1, type=int))
        self.bands_combo.setCurrentIndex(s.value("bands", BAND_COUNTS.index(8),
                                                 type=int))
        self.space_combo.setCurrentIndex(s.value("spacing", 0, type=int))
        for w in (self.display_combo, self.bands_combo, self.space_combo):
            w.blockSignals(False)

    def closeEvent(self, e):
        self.audio.stop()
        s = QSettings("chladni", "chladni-plate")
        s.setValue("geometry", self.saveGeometry())
        s.setValue("shape", self.plate.shape)
        s.setValue("m", self.plate.manual[0])
        s.setValue("n", self.plate.manual[1])
        s.setValue("sign", self.plate.manual[2])
        s.setValue("grains", self.count_spin.value())
        s.setValue("cmap", self.cmap_combo.currentIndex())
        s.setValue("loop", self.loop_chk.isChecked())
        s.setValue("autoadv", self.autoadv_chk.isChecked())
        s.setValue("reactive", self.react_chk.isChecked())
        s.setValue("beats", self.beat_chk.isChecked())
        s.setValue("spectrum", self.spec_chk.isChecked())
        s.setValue("snap", self.snap_chk.isChecked())
        s.setValue("display", self.display_combo.currentIndex())
        s.setValue("bands", self.bands_combo.currentIndex())
        s.setValue("spacing", self.space_combo.currentIndex())
        super().closeEvent(e)


STYLE = """
QWidget { background-color: #101216; color: #d7dae0; font-size: 13px; }
QGroupBox { border: 1px solid #2a2e36; border-radius: 8px;
            margin-top: 14px; padding: 10px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8ab4f8; }
QSlider::groove:horizontal { height: 4px; background: #2a2e36; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -6px 0; border-radius: 7px; background: #8ab4f8; }
QSlider::sub-page:horizontal { background: #3b5ca8; border-radius: 2px; }
QPushButton { background: #1d2129; border: 1px solid #2a2e36; border-radius: 6px; padding: 6px 10px; }
QPushButton:hover { background: #262b36; }
QPushButton:disabled { color: #5a5f68; }
QComboBox, QSpinBox { background: #1d2129; border: 1px solid #2a2e36; border-radius: 6px; padding: 3px 6px; }
QComboBox:disabled, QSpinBox:disabled { color: #5a5f68; }
QLabel#hint { color: #7f858e; }
QListWidget { background: #171a20; border: 1px solid #2a2e36; border-radius: 6px; }
QListWidget::item { padding: 2px 6px; }
QListWidget::item:selected { background: #3b5ca8; }
QProgressBar { background: #1d2129; border: 1px solid #2a2e36; border-radius: 5px; }
QProgressBar::chunk { background: #4caf7d; border-radius: 4px; }
QStatusBar::item { border: none; }
QStatusBar QLabel { padding: 0 8px; }
QScrollArea { background: transparent; }
"""


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
