"""
TFGating is a TensorFlow-based implementation of Spectral Gating, an algorithm for denoising audio signals
================================================
Documentation is available in the docstrings and
online at https://github.com/noisereduce/TensorFlowGating/blob/main/README.md.

Contents
--------
tfgating imports all the functions from PyTorch, and in addition provides:
 TFGating       --- A TensorFlow module that applies a spectral gate to an input signal

The "run.py" script provides a command-line interface for applying the SpectralGate algorithm to audio files.
 CLI command       --- torchgating input_path

Public API in the main TFGating namespace
--------------------------------------
::
 __version__       --- TFGating version string

References
--------------------------------------
The algorithm was originally proposed by Sainburg et al [1] and was previously implemented in a GitHub repository [2]

[1] Sainburg, Tim, and Timothy Q. Gentner. “Toward a Computational Neuroethology of Vocal Communication:
From Bioacoustics to Neurophysiology, Emerging Tools and Future Directions.”

[2] Sainburg, T. (2019). noise-reduction. GitHub. Retrieved from https://github.com/timsainb/noisereduce.
"""


from .tfgating import TFGating
from .version import __version__
