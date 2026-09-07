#!/usr/bin/env python
"""
Build a --curve CSV (rpm,torque at the model's control rate) for generate.py.

Rather than hand-crafting a synthetic drive cycle, this extracts the real
RPM/torque channels straight out of one of your actual 4-channel
recordings, using the exact same channel-extraction and block-averaging
logic (engine_ddsp.data.load_multichannel / block_average) that
preprocess.py used to build the training set -- so the result is
guaranteed consistent with what the model actually saw during training.

python make_curve.py --config config.yaml \
    --input ~/Documents/st8_engine/procedural-engine-sounds/dataset/audio/A_full_set/004_Engine-A.wav \
    --out my_drive_cycle.csv
"""
import argparse
import yaml
from engine_ddsp.data import load_multichannel, block_average


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--input", required=True, help="a real 4-channel recording")
    p.add_argument("--out", default="my_drive_cycle.csv")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    pp = config["preprocess"]

    raw = load_multichannel(args.input, pp["sampling_rate"])
    rpm_raw = raw[pp["rpm_channel"]]
    torque_raw = raw[pp["torque_channel"]]

    rpm = block_average(rpm_raw, pp["block_size"]) * pp["rpm_scale"] + pp["rpm_offset"]
    torque = block_average(torque_raw, pp["block_size"]) * pp["torque_scale"] + pp["torque_offset"]

    with open(args.out, "w") as f:
        f.write("rpm,torque\n")
        for r, t in zip(rpm, torque):
            f.write(f"{r},{t}\n")

    control_rate = pp["sampling_rate"] / pp["block_size"]
    print(f"wrote {len(rpm)} rows ({len(rpm) / control_rate:.1f}s at {control_rate:.1f}Hz) to {args.out}")


if __name__ == "__main__":
    main()
