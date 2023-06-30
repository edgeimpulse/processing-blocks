import argparse
import json
import sys
import os

import numpy as np
from itertools import combinations
import scipy.spatial.distance as distance

sys.path.append('/')
import common.spectrum as spectrum
from common.errors import ConfigurationError

DEFAULT_MAX_NFFT = 1024
DEFAULT_MIN_NFFT = 32
TRANSIENT_THRESHOLD = 0.3
DEBUG_OUTPUT_LEVEL = 0

MODULE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third_party', 'speechpy', '__init__.py')
MODULE_NAME = 'speechpy'
import importlib
import sys
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
speechpy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = speechpy
spec.loader.exec_module(speechpy)


def get_power_spectrum(s, sampling_frequency, fft_length,
     implementation_version=3, frame_length=None, frame_stride=None, take_log=True):

    if frame_length is None:
        frame_length = fft_length / sampling_frequency

    if frame_stride is None:
        frame_stride = frame_length / 2

    if implementation_version < 3:
        raise ConfigurationError('implementation_version should be 3 or higher')

    numframes, _, __ = speechpy.processing.calculate_number_of_frames(
        s,
        implementation_version=implementation_version,
        sampling_frequency=sampling_frequency,
        frame_length=frame_length,
        frame_stride=frame_stride,
        zero_padding=False)

    if (numframes < 1):
        raise ConfigurationError(
            'Frame length is larger than your window size')

    frames = speechpy.processing.stack_frames(
        s,
        implementation_version=implementation_version,
        sampling_frequency=sampling_frequency,
        frame_length=frame_length,
        frame_stride=frame_stride,
        filter=lambda x: np.ones((x,)),
        zero_padding=False)

    power_spectrum = speechpy.processing.power_spectrum(frames, fft_length)

    if take_log:
        power_spectrum = np.clip(power_spectrum, 1e-30, None)
        power_spectrum = 10 * np.log10(power_spectrum)

    return power_spectrum


def detect_transient(specgram):
    assert(specgram.ndim == 2)
    diffs = []
    for i in range(specgram.shape[0] - 1):
        diff = specgram[i + 1, :] - specgram[i, :]
        spec_flux = np.dot(diff, diff) / np.dot(specgram[i, :], specgram[i, :])
        spec_flux = np.sqrt(spec_flux)
        diffs.append(spec_flux)
        if spec_flux > TRANSIENT_THRESHOLD:
            return True
    if DEBUG_OUTPUT_LEVEL > 0:
        max_flux = np.max(diffs)
        print(f'max flux : {max_flux}')
    return False


def autotune_params(data, options):
    '''
    performs autotuning of parameters; the wrapper script calls this
    '''
    window_size_ms = options['window_size_ms'] if 'window_size_ms' in options else None

    frame_stride = 0.01
    frame_length = 0.02
    noise_floor_db = -52
    min_nfft = DEFAULT_MIN_NFFT
    max_nfft = DEFAULT_MAX_NFFT

    # find scaling factor

    max_val = 0
    min_len = 100000000
    for x, y, _ in data:
        max_val = max(max_val, np.max(np.abs(x)))
        min_len = min(min_len, x.shape[0])

    if min_len < min_nfft:
        raise ConfigurationError('Input data is too short')

    nfft = min_nfft
    best = None
    scores = {}
    noise_floors = {}
    while nfft <= min(max_nfft, x.shape[0]):
        win_size = nfft # win_size is frame length in samples
        while win_size <= min(4 * nfft, x.shape[0]):
            sg_class = {}
            class_cnt = {}
            noise_floor_db = 0
            for x, y, interval_ms in data:
                nax = 1 if x.ndim == 1 else x.shape[1]
                x = x.reshape(x.shape[0], nax)
                x = x[:min_len, :]
                fs = 1000 / interval_ms

                if np.any((x < -1) | (x > 1)):
                    x = (x / 2**15).astype(np.float32)

                try:
                    ps_all_axis = None
                    for ax in range(nax):
                        ps = get_power_spectrum(x[:, ax], fs, nfft, frame_length=win_size/fs)
                        floor = np.nanpercentile(ps, 75)
                        noise_floor_db = min(noise_floor_db, floor)
                        if ps_all_axis is None:
                            ps_all_axis = ps
                        else:
                            ps_all_axis = np.concatenate((ps_all_axis, ps), axis=1)

                    if y not in sg_class:
                        sg_class[y] = ps_all_axis
                        class_cnt[y] = 1
                    else:
                        sg_class[y] += ps_all_axis
                        class_cnt[y] += 1
                except Exception as e:
                    print(e)

            for y in sg_class:
                sg_class[y] /= class_cnt[y]

            res = list(combinations(sg_class.keys(), 2))
            if len(res) == 0:
                raise ConfigurationError('At least two classes are required')

            score = 0
            for p in res:
                a = sg_class[p[0]]
                b = sg_class[p[1]]
                score += 1 - distance.cosine(a.flatten(), b.flatten())
            score /= len(res)

            scores[(nfft, win_size)] = score
            noise_floors[(nfft, win_size)] = noise_floor_db

            win_size *= 2
        nfft *= 2

    best = max(scores, key=scores.get)
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    if DEBUG_OUTPUT_LEVEL > 0:
        print('best score: ', scores)

    # second pass, decide whether to overlap
    transient = False
    for x, y, interval_ms in data:
        nax = 1 if x.ndim == 1 else x.shape[1]
        x = x.reshape(x.shape[0], nax)
        x = x[:min_len, :]
        fs = 1000 / interval_ms

        if np.any((x < -1) | (x > 1)):
            x = (x / 2**15).astype(np.float32)

        try:
            for ax in range(nax):
                ps = get_power_spectrum(x[:, ax], fs, best[0], frame_length=best[1]/fs, take_log=False)
                transient = transient or detect_transient(ps)
        except Exception as e:
            print(e)

    frame_length = best[1] / fs
    frame_stride = frame_length * 0.75 if transient else frame_length * 2

    # make sure frame_stride creates no more than 500 frames (api max)
    frame_stride = spectrum.cap_frame_stride(window_size_ms, frame_stride)

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
            'key': 'fft_length',
            'value': best[0]
        },
        {
            'key': 'noise_floor_db',
            'value': np.floor(noise_floors[best])
        }
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto tuning for spectrogram block')
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
