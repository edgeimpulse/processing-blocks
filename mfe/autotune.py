import argparse
import json
import sys
import os
import numpy as np

sys.path.append('/')
import common.spectrum

DEFAULT_FRAME_LENGTH = 0.025 # 25ms
DEFAULT_FRAME_STRIDE = 0.01 # 10ms
DEFAULT_NFFT = 256  # for 8kHz sampling rate
DEFAULT_LOW_FREQUENCY = 80
DEFAULT_NUM_FILTERS = 40
DEFAULT_NOISE_FLOOR_DB = -52

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
    noise_floor_db = -52
    low_frequency = DEFAULT_LOW_FREQUENCY
    implementation_version = 4

    fft_length, num_filters = common.spectrum.audio_set_params(frame_length, data.fs)

    for x, _, _ in data:
        x = x if x.ndim == 1 else x[:, 0]
        padding = data.max_len - x.shape[0]
        if padding > 0:
            x = np.pad(x, (0, padding), 'constant')

        x = (x / 2**15).astype(np.float32)
        x = speechpy.processing.preemphasis(x, cof=0.98, shift=1)
        mfe, _, _ = speechpy.feature.mfe(x, sampling_frequency=data.fs, implementation_version=implementation_version,
                                        frame_length=frame_length,
                                        frame_stride=frame_stride, num_filters=num_filters, fft_length=fft_length,
                                        low_frequency=low_frequency, high_frequency=None)
        mfe = np.clip(mfe, 1e-30, None)
        mfe = 10 * np.log10(mfe)

        floor = np.nanpercentile(mfe, 50)
        noise_floor_db = min(noise_floor_db, floor)

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
            'value': num_filters
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
            'key': 'noise_floor_db',
            'value': np.floor(noise_floor_db)
        }
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto tuning for MFE block')
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
