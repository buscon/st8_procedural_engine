"""
Differentiable synthesis primitives: a harmonic (additive) oscillator bank,
a filtered-noise synthesizer, and a multi-resolution spectral loss.

These nine functions/classes are adapted, with attribution, from
acids-ircam/ddsp_pytorch (MIT License, Copyright (c) Antoine Caillon /
IRCAM), itself an implementation of Engel et al., "DDSP: Differentiable
Digital Signal Processing" (ICLR 2020). Only the generic synthesis math is
kept here -- nothing that depends on audio-domain pitch/loudness extraction
(the original repo's `extract_pitch`/`extract_loudness`, which pull in
`crepe`/TensorFlow and `librosa` respectively). This module has no
dependency beyond `torch`, on purpose, so importing it never drags in
anything unrelated to the actual synthesizer.
"""
import math
import torch
import torch.nn as nn
import torch.fft as fft


def safe_log(x):
    return torch.log(x + 1e-7)


def multiscale_fft(signal, scales, overlap):
    """Multi-resolution STFT magnitude, used for the reconstruction loss."""
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def upsample(signal, factor):
    """Linear-interpolate a per-frame control signal up to audio rate."""
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes, f0, sampling_rate):
    """Zero out harmonic amplitudes whose frequency would alias past Nyquist."""
    n_harm = amplitudes.shape[-1]
    freqs = f0 * torch.arange(1, n_harm + 1).to(f0)
    mask = (freqs < sampling_rate / 2).float() + 1e-4
    return amplitudes * mask


def scale_function(x):
    """Maps an unconstrained network output to a non-negative amplitude."""
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7


def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + n_layers * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_inputs, hidden_size):
    return nn.GRU(n_inputs * hidden_size, hidden_size, batch_first=True)


def harmonic_synth(f0, amplitudes, sampling_rate):
    """Sum of sinusoids at f0 and its harmonics, weighted by `amplitudes`."""
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * f0 / sampling_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal


def amp_to_impulse_response(amp, target_size):
    """Turns predicted per-band magnitudes into a time-domain FIR filter."""
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]
    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)
    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)
    return amp


def fft_convolve(signal, kernel):
    """Fast convolution of a noise signal with a (time-varying) FIR filter."""
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]
    return output


class Reverb(nn.Module):
    """A small learned exponentially-decaying impulse response, applied by
    FFT convolution. Stands in for exhaust/cabin resonance."""

    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))
        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)
        return x
