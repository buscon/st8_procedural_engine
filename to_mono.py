"""
Extract two channels from each multichannel file in a folder and average
them into a standalone mono WAV file (recursive, preserves subfolders).

Usage:
    python to_mono.py --input_dir /path/to/4ch_files --output_dir /path/to/mono_out
    python to_mono.py --input_dir ... --output_dir ... --extension wav --channels 0 1

Requires only: pip install soundfile numpy
"""
import argparse
import pathlib
import soundfile as sf


def convert_file(src, dst, channels):
    x, sr = sf.read(src, always_2d=True)  # (n_samples, n_channels)
    if x.shape[1] <= max(channels):
        raise ValueError(
            f"{src} has only {x.shape[1]} channel(s), "
            f"need at least {max(channels) + 1} for channels={channels}"
        )
    mono = x[:, channels].mean(axis=1)
    sf.write(dst, mono, sr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="folder containing the multichannel files")
    p.add_argument("--output_dir", required=True, help="where to write the mono WAVs")
    p.add_argument("--extension", default="wav", help="file extension to look for (default: wav)")
    p.add_argument("--channels", type=int, nargs=2, default=[0, 1],
                    help="indices of the two channels to average into mono (default: 0 1)")
    args = p.parse_args()

    in_dir = pathlib.Path(args.input_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.rglob(f"*.{args.extension}"))
    if not files:
        print(f"no .{args.extension} files found under {in_dir}")
        return

    for f in files:
        rel = f.relative_to(in_dir)
        dst = out_dir / rel
        dst = dst.with_suffix(".wav")  # always write WAV regardless of source extension
        dst.parent.mkdir(parents=True, exist_ok=True)
        convert_file(f, dst, args.channels)
        print(f"{f} -> {dst}")

    print(f"done: {len(files)} file(s) converted")


if __name__ == "__main__":
    main()
