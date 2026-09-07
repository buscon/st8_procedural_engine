#!/usr/bin/env python
"""
Synthesize engine audio from an RPM/torque trajectory -- no audio input
needed at inference time.

Two ways to provide that trajectory:

1. From one of your real 4-channel recordings (rpm/torque get extracted
   directly, the same way preprocess.py did for training):
   python generate.py --config runs/my_engine/config.yaml \\
       --ckpt runs/my_engine/state.pth \\
       --input4ch path/to/004_Engine-A.wav --out generated.wav

2. From a hand-made CSV with "rpm,torque" columns at the control rate the
   model was trained with (sampling_rate / block_size) -- for a synthetic
   drive cycle you constructed yourself:
   python generate.py --config runs/my_engine/config.yaml \\
       --ckpt runs/my_engine/state.pth \\
       --curve my_drive_cycle.csv --out generated.wav
"""
import argparse
import csv
import yaml
import numpy as np
import torch
import soundfile as sf
from engine_ddsp.model import EngineDDSP
from engine_ddsp.data import load_multichannel, block_average


def curve_from_csv(path):
    rpm, torque = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            rpm.append(float(row["rpm"]))
            torque.append(float(row["torque"]))
    return np.array(rpm, dtype=np.float32), np.array(torque, dtype=np.float32)


def curve_from_4ch(path, preprocess_config):
    """Extract rpm/torque straight out of a real 4-channel recording, using
    the exact same channel-extraction and block-averaging logic
    preprocess.py used to build the training set."""
    pp = preprocess_config
    raw = load_multichannel(path, pp["sampling_rate"])
    rpm_raw = raw[pp["rpm_channel"]]
    torque_raw = raw[pp["torque_channel"]]
    rpm = block_average(rpm_raw, pp["block_size"]) * pp["rpm_scale"] + pp["rpm_offset"]
    torque = block_average(torque_raw, pp["block_size"]) * pp["torque_scale"] + pp["torque_offset"]
    return rpm.astype(np.float32), torque.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input4ch", help="a real 4-channel recording -- rpm/torque are extracted directly from it")
    source.add_argument("--curve", help="CSV with rpm,torque columns at the control rate (for a synthetic drive cycle)")
    p.add_argument("--out", default="generated.wav")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EngineDDSP(**config["model"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    if args.input4ch:
        rpm, torque = curve_from_4ch(args.input4ch, config["preprocess"])
        print(f"extracted rpm/torque from {args.input4ch} ({len(rpm)} control frames)")
    else:
        rpm, torque = curve_from_csv(args.curve)
        print(f"loaded rpm/torque from {args.curve} ({len(rpm)} control frames)")

    cylinders = config["preprocess"]["cylinders"]
    strokes_per_cycle = config["preprocess"]["strokes_per_cycle"]
    f0 = (rpm / 60.0) * (cylinders / strokes_per_cycle)

    mean_rpm, std_rpm = config["data"]["mean_rpm"], config["data"]["std_rpm"]
    mean_torque, std_torque = config["data"]["mean_torque"], config["data"]["std_torque"]

    f0_t = torch.from_numpy(f0).reshape(1, -1, 1).to(device)
    rpm_t = torch.from_numpy((rpm - mean_rpm) / std_rpm).reshape(1, -1, 1).to(device)
    torque_t = torch.from_numpy((torque - mean_torque) / std_torque).reshape(1, -1, 1).to(device)

    with torch.no_grad():
        audio = model(f0_t, rpm_t, torque_t).squeeze().cpu().numpy()

    sf.write(args.out, audio, config["preprocess"]["sampling_rate"])
    print(f"wrote {args.out} ({len(audio) / config['preprocess']['sampling_rate']:.1f}s)")


if __name__ == "__main__":
    main()
