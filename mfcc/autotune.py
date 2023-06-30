import argparse
import json
import sys
import os

import numpy as np

sys.path.append('/')
import common.spectrum

DEFAULT_FRAME_LENGTH = 0.025 # 25ms
DEFAULT_FRAME_STRIDE = 0.02 # 20ms
DEFAULT_LOW_FREQUENCY = 80
DEFAULT_NUM_FILTERS = 32

MODULE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third_party', 'speechpy', '__init__.py')
MODULE_NAME = 'speechpy'
import importlib
import sys
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
speechpy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = speechpy
spec.loader.exec_module(speechpy)


def autotune_params(data, options):

    window_size_ms = options['window_size_ms'] if 'window_size_ms' in options else None
    frame_stride = DEFAULT_FRAME_STRIDE
    # make sure frame_stride creates no more than 500 frames (api max)
    frame_stride = common.spectrum.cap_frame_stride(window_size_ms, frame_stride)

    frame_length = DEFAULT_FRAME_LENGTH
    low_frequency = DEFAULT_LOW_FREQUENCY

    fft_length, _ = common.spectrum.audio_set_params(frame_length, data.fs)

    # 3 seconds https://speechpy.readthedocs.io/en/latest/content/postprocessing.html
    win_size = int(3 / frame_stride) // 2 * 2 + 1

    return [
        {
            'key': 'frame_length',
            'value': np.around(frame_length, 4)
        },
        {
            'key': 'frame_stride',
            'value': np.around(frame_stride, 4)
        },
        {
            'key': 'num_filters',
            'value': 32 # Want to pick a power of 2 so we don't force kissFFT into the binary
        },
        {
            'key': 'fft_length',
            'value': fft_length
        },
        {
            'key': 'low_frequency',
            'value': low_frequency
        },
        {
            'key': 'win_size',
            'value': win_size
        },
        {
            'key': 'num_cepstral',
            'value': 13
        }
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto tuning for MFCC block')
    parser.add_argument('--file', type=str, required=True,
        help='Path to input feature file (numpy file of raw data)')
    parser.add_argument('--out-file', type=str, required=True,
        help='Path to output json file')
    parser.add_argument('--fs', type=str, required=False, default='100',
        help='Sampling rate of the input data')

    args = parser.parse_args()
    fs = int(args.fs)
    out_file = args.out_file

    data = np.load(args.file, allow_pickle=True)

    data = Iter(data, fs)

    opts = {}

    results = autotune_params(data, opts)

    with open(out_file, 'w') as f:
        f.write(json.dumps({ 'success': True, 'results': results } ))
