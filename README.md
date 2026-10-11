# Chladni Plate Simulator v8

Sand on a vibrating square/circular plate settles into Chladni figures —
and the plate can be *played* like an instrument: songs (MP3/OGG/FLAC/WAV)
or a test tone drive the pattern, with every editable frequency band of
the music controlling its own resonance mode.

## Run

    ./run.sh                 # uses .venv if present, else system python
    python chladni_plate.py # equivalent

## Install deps

    python -m pip install -r requirements.txt            # required
    python -m pip install -r requirements-optional.txt   # recommended

(Or regenerate everything with `./make_chladni.sh --install`.)

## Highlights

- Square & circular plates (Bessel modes), momentum sand physics, trails
- Playlist player: beats kick the sand, loudness drives amplitude,
  dominant pitch selects the mode, live spectrum overlay
- EQ-style band console: 2–24 bands, free-drag / ISO octave / ⅓-octave
  presets, note-snapped dividers, per-band gain (Ctrl = fine ×0.1),
  flatten, mute, per-band (m,n) override, JSON layout save/load
- Test tone generator, GIF/PNG export, QSettings persistence

## Keys

←/→ n ±  ·  ↑/↓ m ±  ·  R random  ·  Space pause  ·  T tap  ·  S save PNG
G record GIF  ·  O add songs  ·  A play  ·  N/B next/prev track  ·
Y tone  ·  1/2 plate shape
