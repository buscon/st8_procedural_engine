#!/usr/bin/env python
"""
python train.py --config config.yaml --name my_engine --steps 500000
"""
import argparse
import yaml
import numpy as np
import torch
import soundfile as sf
from os import path, makedirs
from torch.utils.tensorboard import SummaryWriter

from engine_ddsp.model import EngineDDSP
from engine_ddsp.data import Dataset
from engine_ddsp.synth import multiscale_fft, safe_log


@torch.no_grad()
def mean_std(dataloader, index):
    """Running mean/std of one element of each batch tuple (approximate,
    batch-averaged -- fine for normalization purposes)."""
    mean, std, n = 0, 0, 0
    for batch in dataloader:
        x = batch[index]
        n += 1
        mean += (x.mean().item() - mean) / n
        std += (x.std().item() - std) / n
    return mean, std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--name", default="engine_debug")
    p.add_argument("--root", default="runs")
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--start_lr", type=float, default=1e-3)
    p.add_argument("--stop_lr", type=float, default=1e-4)
    p.add_argument("--decay_over", type=int, default=400000)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}"
          f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")

    model = EngineDDSP(**config["model"]).to(device)
    dataset = Dataset(config["preprocess"]["out_dir"])
    dataloader = torch.utils.data.DataLoader(
        dataset, args.batch, shuffle=True, drop_last=True,
    )

    # rpm/torque have arbitrary units and offsets -> normalize them.
    # f0 is left in Hz, unnormalized -- it's the direct oscillator-phase driver.
    mean_rpm, std_rpm = mean_std(dataloader, 2)
    mean_torque, std_torque = mean_std(dataloader, 3)
    config.setdefault("data", {})
    config["data"]["mean_rpm"], config["data"]["std_rpm"] = mean_rpm, std_rpm
    config["data"]["mean_torque"], config["data"]["std_torque"] = mean_torque, std_torque

    run_dir = path.join(args.root, args.name)
    makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir, flush_secs=20)

    with open(path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    opt = torch.optim.Adam(model.parameters(), lr=args.start_lr)

    def lr_at(step):
        # --decay_over is in units of training *steps* (matches --steps),
        # not epochs -- with a few thousand batches per epoch, epoch count
        # never gets anywhere near decay_over, so this must key off `step`.
        if step >= args.decay_over:
            return args.stop_lr
        t = step / args.decay_over
        return args.start_lr * (1 - t) + args.stop_lr * t

    best_loss = float("inf")
    mean_loss, n_element, step = 0, 0, 0
    epochs = int(np.ceil(args.steps / len(dataloader)))

    for e in range(epochs):
        for s, f0, rpm, torque in dataloader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            s = s.to(device)
            f0 = f0.unsqueeze(-1).to(device)
            rpm = rpm.unsqueeze(-1).to(device)
            torque = torque.unsqueeze(-1).to(device)

            rpm_n = (rpm - mean_rpm) / std_rpm
            torque_n = (torque - mean_torque) / std_torque

            y = model(f0, rpm_n, torque_n).squeeze(-1)

            ori_stft = multiscale_fft(s, config["train"]["scales"], config["train"]["overlap"])
            rec_stft = multiscale_fft(y, config["train"]["scales"], config["train"]["overlap"])

            loss = 0
            for s_x, s_y in zip(ori_stft, rec_stft):
                loss = loss + (s_x - s_y).abs().mean() + (safe_log(s_x) - safe_log(s_y)).abs().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            writer.add_scalar("loss", loss.item(), step)
            writer.add_scalar("lr", lr_at(step), step)
            step += 1
            n_element += 1
            mean_loss += (loss.item() - mean_loss) / n_element

        if e % 10 == 0:
            writer.add_scalar("reverb_decay", model.reverb.decay.item(), e)
            writer.add_scalar("reverb_wet", model.reverb.wet.item(), e)

            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(model.state_dict(), path.join(run_dir, "state.pth"))

            mean_loss, n_element = 0, 0

            audio = torch.cat([s, y], -1).reshape(-1).detach().cpu().numpy()
            sf.write(
                path.join(run_dir, f"eval_{e:06d}.wav"),
                audio,
                config["preprocess"]["sampling_rate"],
            )
            print(f"epoch {e}/{epochs}  step {step}  loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
