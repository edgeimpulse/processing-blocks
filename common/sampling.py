import math
import time
import sys
from scipy import signal


def calc_resampled_size(input_sample_rate, output_sample_rate, input_length):
    """Calculate the output size after resampling.
    :returns: integer output size, >= 1
    """
    target_size = int(
        math.ceil((output_sample_rate / input_sample_rate) * (input_length)))
    return max(target_size, 1)


def calculate_freq(interval):
    """ Convert interval (ms) to frequency (Hz)
    """
    freq = 1000 / interval
    if abs(freq - round(freq)) < 0.01:
        freq = round(freq)
    return freq


def calc_decimation_ratios(filter_type, filter_cutoff, fs):
    if filter_type != 'low':
        return 1

    # we support base ratios of 3 and 10 in SDK
    ratios = [3, 10, 30, 100, 1000]
    ratios.reverse()
    for r in ratios:
        if fs / 2 / r * 0.9 > filter_cutoff:
            return r

    return 1


def get_ratio_combo(r):
    if r == 1:
        return [1]
    elif r == 3 or r == 10:
        return [r]
    elif r == 30:
        return [3, 10]
    elif r == 100:
        return [10, 10]
    elif r == 1000:
        return [10, 10, 10]
    else:
        raise ValueError("Invalid decimation ratio: {}".format(r))


def create_decimate_filter(ratio):
    sos = signal.cheby1(8, 0.05, 0.8 / ratio, output='sos')
    zi = signal.sosfilt_zi(sos)
    return sos, zi


def decimate_simple(x, ratio, export=False):
    if x.ndim != 1:
        raise ValueError(f'x must be 1D {x.shape}')
    x = x.reshape(x.shape[0])
    if (ratio == 1):
        return x
    sos, zi = create_decimate_filter(ratio)
    y, zo = signal.sosfilt(sos, x, zi=zi * x[0])
    sl = slice(None, None, ratio)
    y = y[sl]
    if export:
        return y, sos, zi
    return y


class Resampler:
    """ Utility class to handle resampling and logging
    """

    def __init__(self, total_samples):
        self.total_samples = total_samples
        self.ix = 0
        self.last_message = 0

    def resample(self, sample, new_length, original_length):
        # Work out the correct axis
        ds_axis = 0
        if (sample.shape[0] == 1):
            ds_axis = 1

        # Resample
        if (original_length != new_length):
            sample = signal.resample_poly(
                sample, new_length, original_length, axis=ds_axis)

        # Logging
        self.ix += 1
        if (int(round(time.time() * 1000)) - self.last_message >= 3000) or (self.ix == self.total_samples):
            print('[%s/%d] Resampling windows...' %
                  (str(self.ix).rjust(len(str(self.total_samples)), ' '), self.total_samples))

            if (self.ix == self.total_samples):
                print('Resampled %d windows\n' % self.total_samples)

            sys.stdout.flush()
            self.last_message = int(round(time.time() * 1000))

        return sample
