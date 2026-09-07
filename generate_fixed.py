#!/usr/bin/env python
"""
Workaround for a bug in acids-rave's `rave generate`: it decides valid audio
extensions by calling torchaudio.get_audio_backend(), which is a deprecated
no-op in newer torchaudio and returns None -- so rave.core.get_valid_extensions()
falls through all its if/elif branches and returns None too, crashing
get_audio_files() with `TypeError: argument of type 'NoneType' is not iterable`.

This patches only that one function (soundfile's supported extensions,
matching the UserWarning you see about the deprecated backend call) and then
runs the real `generate.main` unmodified -- nothing in the installed
`rave`/`scripts` packages is touched.

Usage (identical to `rave generate`, just via this script instead):
    python generate_fixed.py --model <run_or_ckpt> --input <file_or_folder> \
        --out_path test_generations --gpu 0
"""
import sys
import rave.core

rave.core.get_valid_extensions = lambda: ['.wav', '.flac', '.ogg', '.aiff', '.aif', '.aifc']

from scripts import generate
from absl import app

if __name__ == "__main__":
    sys.argv[0] = "generate"
    app.run(generate.main)
