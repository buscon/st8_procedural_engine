"""
Preprocessing and Dataset for EngineDDSP. Reads multichannel files (audio +
RPM + torque, all at the same sample rate), reduces the audio channels to
mono, block-averages RPM/torque to the model's control rate, and converts
RPM to an instantaneous firing frequency (f0) via the engine's firing order.

Depends only on numpy, soundfile, librosa (for resampling) and torch --
no crepe, no TensorFlow, no third-party DDSP/RAVE package.
"""
import pathlib
import numpy as np
import soundfile as sf
import librosa as li
import torch


def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def load_multichannel(f, sampling_rate):
    """Load every channel of a file and resample all of them to `sampling_rate`."""
    x, sr = sf.read(f, always_2d=True)  # (n_samples, n_channels)
    x = x.astype(np.float32).T  # -> (n_channels, n_samples)
    if sr != sampling_rate:
        x = np.stack([
            li.resample(ch, orig_sr=sr, target_sr=sampling_rate) for ch in x
        ])
    return x


def block_average(x, block_size):
    """Reduce a 1D control signal to one value per block_size-sample block."""
    n_blocks = len(x) // block_size
    x = x[:n_blocks * block_size]
    return x.reshape(n_blocks, block_size).mean(-1)


def preprocess_file(f, sampling_rate, block_size, signal_length, oneshot,
                     audio_channels, rpm_channel, torque_channel,
                     cylinders, strokes_per_cycle,
                     rpm_scale, rpm_offset, torque_scale, torque_offset, **kwargs):
    raw = load_multichannel(f, sampling_rate)  # (n_channels, n_samples)

    audio = raw[audio_channels].mean(0)  # stereo audio channels -> mono
    rpm_raw = raw[rpm_channel]
    torque_raw = raw[torque_channel]

    N = (signal_length - len(audio) % signal_length) % signal_length
    audio = np.pad(audio, (0, N))
    rpm_raw = np.pad(rpm_raw, (0, N))
    torque_raw = np.pad(torque_raw, (0, N))

    if oneshot:
        audio = audio[..., :signal_length]
        rpm_raw = rpm_raw[..., :signal_length]
        torque_raw = torque_raw[..., :signal_length]

    rpm = block_average(rpm_raw, block_size) * rpm_scale + rpm_offset
    torque = block_average(torque_raw, block_size) * torque_scale + torque_offset

    firing_ratio = cylinders / strokes_per_cycle
    f0 = (rpm / 60.0) * firing_ratio  # instantaneous firing frequency, in Hz

    n_frames_per_chunk = signal_length // block_size
    audio = audio.reshape(-1, signal_length)
    f0 = f0.reshape(audio.shape[0], n_frames_per_chunk)
    rpm = rpm.reshape(audio.shape[0], n_frames_per_chunk)
    torque = torque.reshape(audio.shape[0], n_frames_per_chunk)

    return audio, f0, rpm, torque


def run_preprocessing(config):
    from os import makedirs, path
    from tqdm import tqdm

    files = get_files(**config["data"])
    if not files:
        raise FileNotFoundError(
            f"no *.{config['data']['extension']} files found under "
            f"{config['data']['data_location']}"
        )

    signals, f0s, rpms, torques = [], [], [], []
    for f in tqdm(files):
        x, f0, rpm, torque = preprocess_file(f, **config["preprocess"])
        signals.append(x)
        f0s.append(f0)
        rpms.append(rpm)
        torques.append(torque)

    signals = np.concatenate(signals, 0).astype(np.float32)
    f0s = np.concatenate(f0s, 0).astype(np.float32)
    rpms = np.concatenate(rpms, 0).astype(np.float32)
    torques = np.concatenate(torques, 0).astype(np.float32)

    out_dir = config["preprocess"]["out_dir"]
    makedirs(out_dir, exist_ok=True)
    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "f0.npy"), f0s)
    np.save(path.join(out_dir, "rpm.npy"), rpms)
    np.save(path.join(out_dir, "torque.npy"), torques)
    return out_dir


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        from os import path
        self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.f0 = np.load(path.join(out_dir, "f0.npy"))
        self.rpm = np.load(path.join(out_dir, "rpm.npy"))
        self.torque = np.load(path.join(out_dir, "torque.npy"))

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        f0 = torch.from_numpy(self.f0[idx])
        rpm = torch.from_numpy(self.rpm[idx])
        torque = torch.from_numpy(self.torque[idx])
        return s, f0, rpm, torque
