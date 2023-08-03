import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from random import sample
from scipy.stats import shapiro
from scipy.signal import windows, resample_poly
from sklearn.metrics.pairwise import cosine_similarity
import peakutils
import pywt

sys.path.append('/')
from common.spectrum import welch_max_hold, next_power_of_2, zero_handling
from common.wavelet import dwt_features
from common.sampling import calc_decimation_ratios, decimate_simple, get_ratio_combo
from common.errors import log,ConfigurationError

DEFAULT_C = 0.93
DEFAULT_MAX_NFFT = 4096
DEFAULT_MIN_NFFT = 32
SCORE_THRESHOLD = 0.2
DEBUG_OUTPUT_LEVEL = 0


def get_filter_params(dataset, nfft=512):

    spec = {}
    for x, y, interval_ms in dataset:

        fs = 1000 / interval_ms
        nax = x.shape[1]

        for ax in range(nax):
            fx = do_welch(x[:, ax], nfft, fs, True)
            if y not in spec:
                spec[y] = fx
            else:
                spec[y] += fx

    res = list(combinations(spec.keys(), 2))
    if len(res) == 0:
        raise ConfigurationError('At least two classes are required')

    diffs = np.zeros_like(spec[y])
    for p in res:
        a = spec[p[0]]
        b = spec[p[1]]
        diffs += np.abs(a - b)

    diffs = np.pad(diffs, (2,), 'mean')
    peak_x = peakutils.indexes(diffs, thres=0.8, min_dist=3)

    if peak_x.size == 0:
        return 'none', fs / 4, 6, DEFAULT_MIN_NFFT, fs

    y = diffs[peak_x[0]]
    low = max(0, peak_x[0] - 2)
    while low > 2 and diffs[low + 2] > y * 0.5:
        low -= 1

    y = diffs[peak_x[-1]]
    high = min(len(diffs) - 4, peak_x[-1] - 2)
    while high < len(diffs) - 4 and diffs[high + 2] > y * 0.5:
        high += 1

    def bin2hz(x):
        return fs / nfft * x

    if low < 4 and high > len(diffs) - 4 - 3:
        type = 'none'
        freq = '0'
    elif low < 4:
        type = 'low'
        freq = bin2hz(high + 2)
    elif high > len(diffs) - 4 - 3:
        type = 'high'
        freq = bin2hz(low - 2)
    else:
        # bandpass is not impletmented yet
        # type = 'band'
        # freq = str(bin2hz(low)) + ',' + str(bin2hz(high))
        type = 'low'
        freq = bin2hz(high + 2)

    # we need a FFT size that is contains the filtered signal
    if type == 'low':
        min_nfft = nfft / (high + 2) * 2
        min_nfft = next_power_of_2(int(min_nfft))
        min_nfft = max(min_nfft, DEFAULT_MIN_NFFT)
    else:
        min_nfft = DEFAULT_MIN_NFFT

    if DEBUG_OUTPUT_LEVEL > 0:
        print('best filter: ', type, freq, min_nfft)

    return type, freq, 6, min_nfft, fs


def gen_window(n, window_type='boxcar'):
    if window_type == 'hann':
        return windows.hann(n)
    elif window_type == 'boxcar':
        return np.ones(n)
    elif window_type == 'hamming':
        return windows.hamming(n)
    elif window_type == 'blackmanharris':
        return windows.blackmanharris(n)
    elif window_type == 'tukey':
        return windows.tukey(n)
    else:
        raise ValueError('Unknown window type')


def do_welch(X, nfft, fs, use_log, w='boxcar'):
    '''
    returns the MAX_HOLD PSD of the signal X, flattened to a 1D array
    '''
    if X.ndim == 2:
        out = np.array([])
        for dim in range(X.shape[1]):
            win = gen_window(X.shape[0], w)
            x = X[:, dim] * win
            _, fx = welch_max_hold(x, fs, nfft, nfft // 2)
            fx = fx[1:-1]
            if use_log:
                fx = np.log(zero_handling(fx))
            out = np.append(out, fx)
    elif X.ndim == 1:
        win = gen_window(X.shape[0], w)
        x = X * win
        _, fx = welch_max_hold(x, fs, nfft, nfft // 2)
        fx = fx[1:-1]
        if use_log:
            fx = np.log(zero_handling(fx))
        out = fx
    else:
        raise ValueError('X must be 1D or 2D')
    return out


def diff_psd(a, b):
    cs = cosine_similarity([a], [b])
    return 1 - cs[0][0]


def find_key_for_max_val(dic):
    max_value = max(dic.values())  # maximum value
    max_keys = [k for k, v in dic.items() if v == max_value]
    return max_keys


def find_best_nfft(dataset, max_nfft=DEFAULT_MAX_NFFT, min_nfft=DEFAULT_MIN_NFFT,
                C=DEFAULT_C, use_log=True, to_plot=False, decimate_factor=1):
    '''
    @param C: factor to penalize large nfft (default: 0.9), larger nfft
            results in more complex models and requires more resources
    returns the best nfft for the dataset
    '''

    data = {}
    ignored = 0
    for x, y, interval_ms in dataset:

        fs = 1000 / interval_ms

        if decimate_factor > 1:
            x = resample_poly(x, 1, decimate_factor)

        # length has to be at least min_nfft + 1 to have sizes to compare
        if (x.shape[0] <= min_nfft):
            ignored += 1
            continue

        if y not in data:
            data[y] = {}

        if (x.shape[0] * 2 <= min_nfft):
            raise ConfigurationError(f'Signal {x.shape[0]} is too short to run auto tuning. {min_nfft} samples needed.')
        max_nfft = min(max_nfft, next_power_of_2(x.shape[0]))

        nfft = min_nfft
        while nfft <= max_nfft:
            fx = do_welch(x, nfft, fs, use_log)
            if nfft not in data[y]:
                data[y][nfft] = fx
            else:
                data[y][nfft] += fx
            nfft *= 2

    if ignored > 0:
        print(f'Ignored {ignored} short samples')

    if len(data) == 0:
        raise ValueError('No valid data')

    nfft = min_nfft
    nffts = []
    while nfft <= max_nfft:
        nffts.append(nfft)
        nfft *= 2

    res = list(combinations(data.keys(), 2))
    if len(res) == 0:
        raise ConfigurationError('At least two classes are required')

    diffs = []
    weighted_diffs = []
    for nfft in nffts:
        diff = 0
        for p in res:
            # It's possible that some classes may have only short samples, in which case larger FFTs won't be calculated
            # Thus, need to check before access
            if nfft in data[p[0]] and nfft in data[p[1]]:
                a = data[p[0]][nfft]
                b = data[p[1]][nfft]
                diff += diff_psd(a, b)
        diff /= len(res)
        diffs.append(diff)
        weighted_diffs.append(diff * np.power(C, np.log2(nfft)))

    '''
    print(nffts)
    print(weighted_diffs)
    print(diffs)
    '''
    best_nfft_idx = np.argmax(np.array(weighted_diffs))
    if DEBUG_OUTPUT_LEVEL > 0:
        print('best_nfft is ', nffts[best_nfft_idx], ' ',
              diffs[best_nfft_idx], ' ', weighted_diffs[best_nfft_idx])

    if to_plot:
        fig, ax = plt.subplots()
        ax.plot(nffts, (diffs), 'r+')
        ax.plot(nffts, (weighted_diffs), 'g+')
        ax.plot(nffts[best_nfft_idx], (weighted_diffs[best_nfft_idx]), 'bo')
        ax.set_xscale('log')
        ax.grid()
        plt.show()

    if diffs[best_nfft_idx] > SCORE_THRESHOLD:
        return nffts[best_nfft_idx], diffs[best_nfft_idx]
    else:
        print(f'Unable to find a good FFT size! Returning {nffts[-1]}')
        return nffts[-1], diffs[best_nfft_idx]


def find_best_window(X, Y, fs, use_log=True, nfft=128):
    windows = ['boxcar', 'hamming', 'hann', 'blackmanharris', 'tukey']
    assert(len(X) == len(Y))

    data = {}
    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        if y not in data:
            data[y] = {}

        for w in windows:
            fx = do_welch(x, nfft, fs, use_log, w)
            if w not in data[y]:
                data[y][w] = fx
            else:
                data[y][w] += fx

    res = list(combinations(data.keys(), 2))
    if len(res) == 0:
        raise ConfigurationError('At least two classes are required')

    diffs = []
    for w in windows:
        diff = 0
        for p in res:
            a = data[p[0]][w]
            b = data[p[1]][w]
            diff += diff_psd(a, b)
        diff /= len(res)
        diffs.append(diff)

    # print(diffs)
    best_idx = np.argmax(np.array(diffs))
    if DEBUG_OUTPUT_LEVEL > 0:
        print('best window is ', windows[best_idx])

    return windows[best_idx]


def test_spec_norm(X, fs, nfft=128, plot=False):
    fx_log = do_welch(X, nfft, fs, True)
    fx_lin = do_welch(X, nfft, fs, False)

    _, norm_log = shapiro(fx_log)
    _, norm_lin = shapiro(fx_lin)

    if plot:
        plt.hist(fx_log)
        plt.hist(fx_lin)
        plt.show()

    return norm_log, norm_lin


def should_take_log(X, fs, sample_size=100, nfft=128):
    '''
    returns 1 if the dataset is likely to be log-normal, -1 if it is likely to be linear-normal
    '''
    use_log = 0
    use_lin = 0

    if len(X) <= sample_size:
        Xs = X
    else:
        Xs = sample(X, sample_size)

    for x in Xs:
        norm_log, norm_lin = test_spec_norm(x, fs, nfft)
        if norm_log > norm_lin:
            use_log += 1
        else:
            use_lin += 1

    score = (use_log - use_lin) / (use_log + use_lin)
    if DEBUG_OUTPUT_LEVEL > 0:
        print('log v lin:', score)
    return score > 0, score


def recommend_decimation_ratio(nfft):
    '''
    returns the decimation ratio and fft size that should be used for the dataset
    '''
    if nfft <= 512:
        return 1, 512
    else:
        return int(nfft / 256), 256


def export_wav_filters(filename, wavlets):
    with open(filename, 'w') as f:
        f.write('//generated by autotune.export\n')
        for w in wavlets:
            wl = pywt.Wavelet(w)
            w = w.replace('.', 'p')
            f.write(f'\nstatic const std::array<std::array<float, {len(wl.dec_lo)}>, 2> {w} = {{{{\n')
            f.write('    ' + str(wl.dec_lo).replace('[', '{{').replace(']', '}}'))
            f.write(',\n')
            f.write('    ' + str(wl.dec_hi).replace('[', '{{').replace(']', '}}'))
            f.write('\n}};\n')


def make_wav_list():
    wavlets = pywt.wavelist(kind='discrete')
    same_as_haar = ['db1', 'bior1.1', 'rbio1.1']

    # line = "        else if (strcmp(wav, \"db4\") == 0) get_filter<len>(db4, h, g);"
    selected = []
    for w in wavlets:
        wl = pywt.Wavelet(w)
        if len(wl.dec_lo) > 20 or w in same_as_haar:
            continue
        selected.append(w)

        # print(line.replace("db4", w.replace(".", "p")).replace("len", str(len(wl.dec_lo))))
    return selected


def gather_wav_features(X, w, level):
    if X.ndim == 2:
        out = np.array([])
        for dim in range(X.shape[1]):
            x = X[:, dim]
            fx, _, _ = dwt_features(x, w, level)
            out = np.append(out, fx)
    elif X.ndim == 1:
        fx, _, _ = dwt_features(X, w, level)
        out = fx
    else:
        raise ValueError('X must be 1D or 2D')
    return out


def find_best_wavelet(dataset, sample_size=100, window_size_ms=None):
    wavlets = make_wav_list()

    min_len = 100000000
    for x, _, interval_ms in dataset:
        min_len = min(min_len, x.shape[0])
        if window_size_ms is not None:
            win_size = int(window_size_ms / interval_ms)
            min_len = min(min_len, win_size)

    if min_len < 1:
        raise ConfigurationError('Too few samples to use wavelet')

    level = int(np.log2(min_len)) - 6 # empirical
    level = min(level, 7)
    if level < 1:
        min_len_ms = np.ceil(128 * interval_ms).astype(int)
        print(f'INFO: Skipping wavelets. Window should be at least {min_len_ms} ms or 128 samples for wavelets. Only considering FFT options.')
        raise ConfigurationError('Too few samples to use wavelet')

    data = {}
    cnt = {}
    progress = 0
    num_classes = len(set([y for _, y, _ in dataset]))
    for x, y, _ in dataset:

        x = np.array(x)

        if y not in data:
            data[y] = {}
            cnt[y] = 0
            if DEBUG_OUTPUT_LEVEL > 1:
                print(y)
        else:
            cnt[y] += 1

        if cnt[y] >= sample_size:
            continue

        for w in wavlets:
            fx = gather_wav_features(x[:min_len, :], w, level)
            fx = np.array(fx)
            if w not in data[y]:
                data[y][w] = fx
            else:
                data[y][w] += fx

        progress += 1
        if progress % 20 == 0:
            print(f'[{progress}/{sample_size * num_classes}] Finding best wavelet...')

    res = list(combinations(data.keys(), 2))
    if len(res) == 0:
        raise ConfigurationError('At least two classes are required')

    diffs = []
    for w in wavlets:
        diff = 0
        for p in res:
            a = data[p[0]][w]
            b = data[p[1]][w]
            try:
                cs = cosine_similarity([a], [b])
                diff += 1 - cs[0][0]
            except Exception as e:
                print(e)
        diff /= len(res)
        diffs.append(diff)

    best_idx = np.argmax(np.array(diffs))
    if DEBUG_OUTPUT_LEVEL > 0:
        print(f'best wavelet is {wavlets[best_idx]} at level {level} ({diffs[best_idx]})')

    return wavlets[best_idx], level, diffs[best_idx]


def adjust_nfft_range(min_x_len, max_x_len, min_nfft, max_nfft, decimate_factor):
    if DEBUG_OUTPUT_LEVEL > 0:
        print(f'original nfft range: {min_nfft} - {max_nfft}')
    min_x_len /= decimate_factor
    max_x_len /= decimate_factor
    min_nfft = min(min_nfft, next_power_of_2(int(min_x_len / 2)))
    max_nfft = min(max_nfft, next_power_of_2(int(max_x_len)))
    if DEBUG_OUTPUT_LEVEL > 0:
        print(f'adjusted nfft range: {min_nfft} - {max_nfft}')
    return min_nfft, max_nfft


def autotune_params(data, options):
    '''
    performs autotuning of parameters; the wrapper script calls this
    data: format defined here: studio/dsp-pipeline/common/dataset.py
    '''
    max_nfft = options['max_nfft'] if 'max_nfft' in options else DEFAULT_MAX_NFFT
    penalty_factor = options['penalty_factor'] if 'penalty_factor' in options else DEFAULT_C
    window_size_ms = options['window_size_ms'] if 'window_size_ms' in options else None

    if len(data.y_label_set) < 2:
        raise ConfigurationError('Auto tuning requires 2 or more labeled classes. Please add data for more classes.')

    # find scaling factor and range of sizes
    max_val = 0
    min_len = 5000000
    max_len = 0
    for x, _, _ in data:
        max_val = max(max_val, np.max(np.abs(x)))
        min_len = min(min_len, x.shape[0])
        max_len = max(max_len, x.shape[0])
    scale = 1 / max_val if max_val > 0 else 1

    if max_len <= 16:
        raise ConfigurationError(f'Signals or windows are too short to run auto tuning.  Need at least 16 samples per raw data window. Max was {max_len}.')

    filter_type, filter_cutoff, filter_order, min_nfft, sampling_freq = get_filter_params(data)
    ratio = calc_decimation_ratios(filter_type, filter_cutoff, sampling_freq)
    if min_len / ratio < 16: # don't decimate if it's too short
        ratio = 1
    if DEBUG_OUTPUT_LEVEL > 0:
        print(f'decimation ratio: {ratio} at {sampling_freq}')
    min_nfft, max_nfft = adjust_nfft_range(min_len, max_len, min_nfft, max_nfft, ratio)

    # do_log, _ = should_take_log(X, fs)
    nfft, score_fft = find_best_nfft(data, use_log=True, max_nfft=max_nfft, min_nfft=min_nfft, C=penalty_factor, decimate_factor=ratio)
    # window = find_best_window(X, y, fs, nfft=nfft, use_log=do_log)
    try:
        wavelet, level, score_wavelet = find_best_wavelet(data, window_size_ms=window_size_ms)
    except Exception as e:
        if not isinstance(e, ConfigurationError):
            log('Autotune: find best wavelet failed', e)
        wavelet = None
        level = None
        score_wavelet = 0

    if score_fft > score_wavelet:
        analysis_type = 'FFT'
    else:
        analysis_type = 'Wavelet'
        filter_type = 'none'
        ratio = 1

    return [
        {
            'key': 'scale-axes',
            'value': scale
        },
        {
            'key': 'input-decimation-ratio',
            'value': ratio
        },
        {
            'key': 'fft-length',
            'value': nfft
        },
        {
            'key': 'do-fft-overlap',
            'value': 'false'
        },
        {
            'key': 'filter-type',
            'value': filter_type
        },
        {
            'key': 'filter-cutoff',
            'value': filter_cutoff
        },
        {
            'key': 'filter-order',
            'value': filter_order
        },
        {
            'key': 'analysis-type',
            'value': analysis_type,
        },
        {
            'key': 'wavelet',
            'value': wavelet
        },
        {
            'key': 'wavelet-level',
            'value': level
        }
    ]


class Iter:
    def __init__(self, data, fs):
        self.data = data
        self.ix = 0
        self.interval_ms = 1000 / fs
        self.y_label_set = np.unique(self.data[:,0])

    def reset(self):
        self.ix = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ix >= len(self.data):
            self.reset()
            raise StopIteration

        # Get y data
        y = self.data[self.ix, 0]
        X = self.data[self.ix, 1]
        self.ix += 1

        return X, y, self.interval_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto tuning for spectral analysis block')
    parser.add_argument('--file', type=str, required=True,
        help='Path to input feature file (numpy file of raw data)')
    parser.add_argument('--out-file', type=str, required=True,
        help='Path to output json file')
    parser.add_argument('--fs', type=str, required=False, default='100',
        help='Sampling rate of the input data')
    parser.add_argument('--max-fft-size', type=int, required=False, default=DEFAULT_MAX_NFFT,
        help='maximum fft size to use')
    parser.add_argument('--C', type=float, required=False, default=DEFAULT_C,
        help='Penalty factor for large fft size')

    args = parser.parse_args()

    fs = int(args.fs)
    max_fft_size = args.max_fft_size
    C = args.C
    out_file = args.out_file

    data = np.load(args.file, allow_pickle=True)

    clean_rows = []
    # Filter out samples with NaN
    for row in data:
        if np.isnan(row[1]).any():
            continue
        clean_rows.append(row)

    data = Iter(np.array(clean_rows), fs)

    opts = {
        'max_nfft': max_fft_size,
        'penalty_factor': C
    }

    results = autotune_params(data, opts)

    with open(out_file, 'w') as f:
        f.write(json.dumps({ 'success': True, 'results': results } ))
