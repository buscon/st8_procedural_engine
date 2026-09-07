# engine-ddsp

A standalone DDSP-style synthesizer conditioned on RPM and torque instead
of pitch and loudness. It's self-contained on purpose: no cloned
third-party repo to patch, no unrelated dependencies (no `crepe`, no
TensorFlow, no Flask, no `gin`/`absl`).

The only borrowed code is `engine_ddsp/synth.py` -- nine generic PyTorch
functions (the harmonic oscillator, the filtered-noise synthesizer, the
multi-resolution spectral loss) adapted with attribution from
[acids-ircam/ddsp_pytorch](https://github.com/acids-ircam/ddsp_pytorch)
(MIT License). Everything that pulled in `crepe`/TensorFlow in the
original repo (its audio-based pitch/loudness extraction) is left out
entirely, since RPM and torque replace that role directly.

## How it works

Standard DDSP extracts pitch (f0) and loudness from audio and feeds those
two numbers per frame into a decoder network that predicts synthesizer
parameters. Here, instead of extracting anything, RPM converts directly to
the engine's firing frequency (an f0 in exactly the sense DDSP already
uses), and RPM + torque condition the decoder the way loudness used to.
Mono audio is only ever the *training target* -- at generation time you
need just an RPM/torque curve, no audio at all.

## Install

```
python -m venv .venv && source .venv/bin/activate    # or reuse an existing venv
pip install -r requirements.txt
```

If you're reusing the venv from a torch install you already fought hard to
get working for a specific GPU/CUDA version, just skip installing `torch`
from this file and leave your existing one in place -- nothing else here
cares which torch build you have, only that it works.

## Usage

1. Edit `config.yaml`: `data_location`/`extension` for your files,
   `sampling_rate` (must match your files' real rate), `audio_channels` /
   `rpm_channel` / `torque_channel`, and `cylinders` / `strokes_per_cycle`
   for the RPM-to-firing-frequency formula. `signal_length` must stay an
   exact multiple of `block_size`.

2. Preprocess:
   ```
   python preprocess.py --config config.yaml
   ```

3. Sanity-check before committing to a long run:
   ```python
   import numpy as np
   f0 = np.load("preprocessed/f0.npy")
   rpm = np.load("preprocessed/rpm.npy")
   print(f0.min(), f0.max(), rpm.min(), rpm.max())
   ```
   Values should look like plausible engine numbers, not all zero or NaN.

4. Train:
   ```
   python train.py --config config.yaml --name my_engine --steps 500000
   ```
   Watch it with `tensorboard --logdir runs`; `runs/my_engine/eval_*.wav`
   interleaves real vs. reconstructed audio every 10 epochs.

5. Generate from an RPM/torque curve alone, once trained:
   ```
   python generate.py --config runs/my_engine/config.yaml \
       --ckpt runs/my_engine/state.pth --curve my_drive_cycle.csv --out out.wav
   ```
   `my_drive_cycle.csv` needs `rpm,torque` columns at the control rate
   (`sampling_rate / block_size` -- 100 Hz with the config defaults).

## Verify the firing-order formula

`f0 = (rpm / 60) * (cylinders / strokes_per_cycle)` is the standard
formula, but the constant depends on cylinder count and 2- vs 4-stroke.
Before trusting it through a long training run, plot a spectrogram of a
steady-RPM segment and confirm the harmonic comb lines up with the
predicted f0 (an order-tracking check).

## If your real sound has structure this doesn't capture

Turbo whine, strong non-order-locked resonances, and similar effects sit
outside what a harmonic+noise model captures well. The natural next step
is an auxiliary loss term supervising energy at the expected engine-order
harmonic bins -- see "Physics-Informed Neural Engine Sound Modeling with
Differentiable Pulse-Train Synthesis" (arXiv 2603.09391), which reports a
real improvement in harmonic reconstruction from exactly this.

## License note

`engine_ddsp/synth.py` carries forward the MIT license of the code it's
adapted from (acids-ircam/ddsp_pytorch, Copyright (c) Antoine Caillon).
Everything else in this package is original.
