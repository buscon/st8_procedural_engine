"""
EngineDDSP: a harmonic+noise differentiable synthesizer conditioned on
(f0, rpm, torque) instead of the usual (pitch, loudness). f0 -- derived
from RPM via the engine's firing order -- drives the harmonic oscillator's
phase directly, locking synthesis to engine order. RPM and torque both feed
the decoder as extra conditioning, the role loudness plays in standard
DDSP, so the network can shape harmonic distribution and noise energy as
speed and load change.

Depends only on .synth (pure PyTorch) -- no third-party DDSP/RAVE package
required.
"""
import math
import torch
import torch.nn as nn
from .synth import (mlp, gru, scale_function, remove_above_nyquist, upsample,
                     harmonic_synth, amp_to_impulse_response, fft_convolve,
                     Reverb)


class EngineDDSP(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate, block_size):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        n_inputs = 3  # f0, rpm, torque
        self.in_mlps = nn.ModuleList(
            [mlp(1, hidden_size, 3) for _ in range(n_inputs)]
        )
        self.gru = gru(n_inputs, hidden_size)
        self.out_mlp = mlp(hidden_size + n_inputs, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, f0, rpm, torque):
        hidden = torch.cat([
            self.in_mlps[0](f0),
            self.in_mlps[1](rpm),
            self.in_mlps[2](torque),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], f0, rpm, torque], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part -- f0 drives the oscillator phase
        param = scale_function(self.proj_matrices[0](hidden))
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]
        amplitudes = remove_above_nyquist(amplitudes, f0, self.sampling_rate)
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        f0_up = upsample(f0, self.block_size)

        harmonic = harmonic_synth(f0_up, amplitudes, self.sampling_rate)

        # filtered-noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)
        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0], impulse.shape[1], self.block_size,
        ).to(impulse) * 2 - 1
        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise
        signal = self.reverb(signal)  # stand-in for exhaust/cabin resonance

        return signal

    def realtime_forward(self, f0, rpm, torque):
        """Streaming version for later real-time use (e.g. a live RPM/torque feed)."""
        hidden = torch.cat([
            self.in_mlps[0](f0),
            self.in_mlps[1](rpm),
            self.in_mlps[2](torque),
        ], -1)
        gru_out, cache = self.gru(hidden, self.cache_gru)
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, f0, rpm, torque], -1)
        hidden = self.out_mlp(hidden)

        param = scale_function(self.proj_matrices[0](hidden))
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]
        amplitudes = remove_above_nyquist(amplitudes, f0, self.sampling_rate)
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        f0_up = upsample(f0, self.block_size)

        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * f0_up / self.sampling_rate, 1)
        omega = omega + self.phase
        self.phase.copy_(omega[0, -1, 0] % (2 * math.pi))

        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
        harmonic = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)

        param = scale_function(self.proj_matrices[1](hidden) - 5)
        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0], impulse.shape[1], self.block_size,
        ).to(impulse) * 2 - 1
        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        return harmonic + noise
