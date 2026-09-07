#!/usr/bin/env python
"""
Synthesize engine audio from an RPM/torque curve alone -- no audio input.

python generate.py --config runs/my_engine/config.yaml \\
    --ckpt runs/my_engine/state.pth --curve my_drive_cycle.csv --out generated.wav

my_drive_cycle.csv needs "rpm,torque" columns at the control rate the model
was trained with (sampling_rate / block_size).
"""
import argparse
import csv
import yaml
import numpy as np
import torch
import soundfile as sf
from engine_ddsp.model import EngineDDSP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--curve", required=True)
    p.add_argument("--out", default="generated.wav")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EngineDDSP(**config["model"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    rpm, torque = [], []
    with open(args.curve) as f:
        for row in csv.DictReader(f):
            rpm.append(float(row["rpm"]))
            torque.append(float(row["torque"]))
    rpm = np.array(rpm, dtype=np.float32)
    torque = np.array(torque, dtype=np.float32)

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
