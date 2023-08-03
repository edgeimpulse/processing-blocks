import argparse
import json
import sys
import math
import numpy as np
from scipy import signal
from scipy.fftpack import fft
import peakutils
from scipy.stats import skew
from scipy.stats import kurtosis as calculateKurtosis
from scipy.signal import tf2zpk
import pathlib

sys.path.append('/')
from common.spectrum import welch_max_hold, zero_handling

ROOT = pathlib.Path(__file__).parent
sys.path.append(str(ROOT / '..'))
sys.path.append(str(object=ROOT ))
from common.errors import ConfigurationError
from common.wavelet import dwt_features, get_max_level, get_min_length, get_wavefunc
from common.sampling import decimate_simple, get_ratio_combo


def filter_is_stable(filter_order, filter_cutoff, sampling_freq):
    # This algorithm is based on the C++ implementation
    # It breaks the filter into two stages, but can still go unstable
    M_PI = 3.14159265358979323846264338327950288
    n_steps = int(filter_order / 2)
    a = np.float32(np.tan(M_PI * filter_cutoff / sampling_freq))
    a2 = np.float32(pow(a, 2))
    A = np.empty(n_steps, dtype='float32')
    d1 = np.empty(n_steps, dtype='float32')
    d2 = np.empty(n_steps, dtype='float32')
    poles = []
    for ix in range(n_steps):
        r = np.sin(M_PI * ((2.0 * ix) + 1.0) / (2.0 * filter_order))
        sampling_freq = a2 + (2.0 * a * r) + 1.0
        A[ix] = a2 / sampling_freq
        d1[ix] = 2.0 * (1 - a2) / sampling_freq
        d2[ix] = -(a2 - (2.0 * a * r) + 1.0) / sampling_freq

        _, p, _ = tf2zpk(A[ix] * np.float32([1, 2, 1]), np.float32([1, -d1[ix], -d2[ix]]))
        poles.append(p)

    return np.max(np.abs(poles)) < 1


def create_filter(type, freq_hz, cut_off_freq, filter_order):
    if (filter_order % 2 != 0):
        raise ConfigurationError('Filter order needs to be even (2, 4, 6, 8)')
    if (filter_order < 1 or filter_order > 9):
        raise ConfigurationError('Filter order needs to be between 2 and 8')
    # Normalized frequency (6 / 62.5) = 0.096
    Wn = cut_off_freq * 2 / freq_hz

    # Catch when frequency too low
    if (Wn >= 1.0):
        raise ConfigurationError('Cut-off frequency is above Nyquist (1/2 sample rate (' +
                        str(freq_hz/2)+')) Choose lower cutoff frequency.')

    return signal.butter(filter_order, Wn=Wn, btype=type, output='sos')


def frequency_domain_graph(sampling_freq, x):
    N = len(x)
    fx = 2.0/N * np.abs(x[0:N//2])
    return fx.tolist()


def frequency_domain_graph_y(sampling_freq, lenX):
    N = lenX
    T = 1 / sampling_freq
    freq_space = np.linspace(0.0, 1.0/(2.0*T), N//2)
    return freq_space.tolist()


def spectral_power_graph(sampling_freq, x, n_fft):
    tx, Pxx_denx = signal.periodogram(x, sampling_freq, nfft=n_fft)
    return tx.tolist(), Pxx_denx[1:].tolist()


def find_peaks_in_fft(sampling_freq, x, threshold, count):
    N = len(x)
    T = 1 / sampling_freq

    # yes, this is all redundant but my Python is shit
    freq_space = np.linspace(0.0, 1.0/(2.0*T), N//2)
    vx = 2.0/N * np.abs(x[0:N//2])

    px = []

    # find and draw all peaks on the graph
    peaks_x = peakutils.indexes(vx, thres=0)
    for p in peaks_x:
        if (x[p] < threshold):
            continue
        px.append([ freq_space[p], vx[p] ])

    # take the top three peaks from every
    px = sorted(px, key=lambda x: x[1], reverse=True)[0:count]

    # if length not good enough add [0,0]
    for k in range(len(px), count):
        px.append([ 0, 0 ])

    return px


def calculate_spectral_power_edges(sampling_freq, x, edges, n_fft):
    fx, Pxx_denx = signal.periodogram(x, sampling_freq, nfft=n_fft)

    def calculate_edges(f, Pxx_den):
        power_data = []

        for k in range(0, len(edges) - 1):
            sum = 0
            total = 0

            for px in range(0, len(Pxx_den)):
                if (f[px] >= edges[k] and f[px] < edges[k + 1]):
                    sum += Pxx_den[px]
                    total += 1

            if total == 0:
                power_data.append(0)
            else:
                power_data.append(sum / total)

        return power_data

    return calculate_edges(fx, Pxx_denx)


def add_stats_features(fx, features, labels, s):
    features.append(np.sqrt(np.mean(np.square(fx))))
    labels.append('RMS' + s)
    features.append(float(np.std(fx)))
    labels.append('Stdev' + s)
    features.append(float(skew(fx)))
    labels.append('Skewness' + s)
    features.append(float(calculateKurtosis(fx)))
    labels.append('Kurtosis' + s)
    return features, labels


def decimate(x, decimate_factor):
    assert(decimate_factor >= 1)
    assert(type(decimate_factor) == int)
    if (decimate_factor == 1):
        return x
    elif (decimate_factor < 64):
        return decimate_simple(x, decimate_factor)
    else:
        current_ratio = 1
        while decimate_factor >= current_ratio * 16:
            x = decimate_simple(x, 16)
            current_ratio *= 16
        return signal.resample_poly(x, current_ratio, decimate_factor)


def extract_spec_features(fx, sampling_freq, fft_length, filter_type, filter_cutoff,
                          do_log, do_fft_overlap, spec_stats, suffix=''):
    '''
    Extracts spectral features from the given signal.
    :param spec_stats: Extract spectral statistics (skewness, kurtosis)
    :param auto_downsample: Downsample if low pass filter is in use
    '''

    #print(f'extract_spec_features: {sampling_freq}, {input_decimation_ratio}, {suffix}')

    features = []

    # add root mean square of the features
    features.append(np.sqrt(np.mean(np.square(fx))))
    # add labels as well
    labels = [ 'RMS' + suffix ]

    # When mean is zero (subtracted out above), stdev == rms
    # Intentionally skip std dev
    features.append(float(skew(fx)))
    labels.append('Skewness' + suffix)
    features.append(float(calculateKurtosis(fx)))
    labels.append('Kurtosis' + suffix)

    freqs, spec_powers = welch_max_hold(
        fx, sampling_freq, nfft=fft_length, n_overlap=fft_length/2 if do_fft_overlap else 0)

    if spec_stats:
        features.append(float(skew(spec_powers)))
        labels.append('Spectral Skewness' + suffix)
        features.append(float(calculateKurtosis(spec_powers)))
        labels.append('Spectral Kurtosis' + suffix)

    freq_spacing = freqs[1]
    if do_log:
        spec_powers = np.log10(zero_handling(spec_powers))

    fft_band_labels = []

    # Optimization: since we subtract the mean at the begining, bin 0 (DC) will always be ~0, so skip it
    for i in range(1, len(freqs)):
        # low-pass filter? skip everything > cutoff
        if (filter_type == 'low' and freqs[i] - freq_spacing/2 > filter_cutoff):
            break  # no more interesting bins
        # high-pass filter? skip everything < cutoff
        if (filter_type == 'high' and freqs[i] + freq_spacing/2 < filter_cutoff):
            continue

        features.append(spec_powers[i])

        band_from = str(round(freqs[i] - freq_spacing/2, 2))
        band_to = str(round(freqs[i] + freq_spacing/2, 2))
        fft_band_labels.append(
                               'Spectral Power ' + band_from + ' - ' + band_to + ' Hz')

    if not fft_band_labels:
        raise ConfigurationError("Cutoff frequency masked all FFT bins.")

    labels += fft_band_labels
    return features, labels, spec_powers.tolist(), freqs.tolist()


def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes, input_decimation_ratio,
                      filter_type, filter_cutoff, filter_order, analysis_type, fft_length, spectral_peaks_count,
                      spectral_peaks_threshold, spectral_power_edges, do_log, do_fft_overlap,
                      wavelet_level, wavelet, extra_low_freq):
    if (implementation_version != 1 and implementation_version != 2
        and implementation_version != 3 and implementation_version != 4):
        raise Exception('implementation_version should be 1, 2, 3 or 4')

    if (not math.log2(fft_length).is_integer()):
        raise ConfigurationError('FFT length must be a power of 2')

    # reshape first
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    if implementation_version == 1 and isinstance(spectral_power_edges, str):
        spectral_power_edges = [float(item.strip())
                                for item in spectral_power_edges.split(',')]

    features = []
    graphs = []
    labels = []

    after_filter_graph = {}
    after_filter_size = 0
    freq_domain_graph = {}
    spect_power_graph = {}
    spect_power_y = []
    butter_sos = None
    wavelet_graph = {}

    input_decimation_ratio = int(input_decimation_ratio)
    if input_decimation_ratio > 1:
        if implementation_version < 4:
            raise ConfigurationError(
                'Input decimation is only supported in implementation version 4')
        nyquist = sampling_freq / 2 / input_decimation_ratio
        if (filter_type == 'low' or filter_type == 'high') and filter_cutoff > nyquist * 0.9:
            raise ConfigurationError(
                'decimation ratio is too high, please lower the input decimation ratio')

    # create filter to denoise
    if (filter_type == 'low' or filter_type == 'high') and filter_order > 0:
        butter_sos = create_filter(
            filter_type, sampling_freq / input_decimation_ratio, filter_cutoff, filter_order)
        butter_sos = np.float32(butter_sos)

        # Design check (Python only step)
        if not filter_is_stable(filter_order, filter_cutoff, sampling_freq / input_decimation_ratio):
            raise ConfigurationError(
                'Filter created is not stable. Move cutoff away from 0 or Nyquist')

    lf_features = []
    lf_labels = []

    original_freq = sampling_freq

    for ax in range(0, len(axes)):
        fx = raw_data[:, ax]

        # potentially scale data from sensor
        fx = np.array(fx, dtype='f') * scale_axes

        sampling_freq = original_freq

        if input_decimation_ratio > 1:
            ratios = get_ratio_combo(input_decimation_ratio)
            for ratio in ratios:
                fx = decimate_simple(fx, ratio)
                sampling_freq /= ratio

        # apply filter to denoise
        if butter_sos is not None:
            fx = signal.sosfilt(butter_sos, fx)

        # offset by the mean
        fx = np.array(fx) - np.mean(fx)

        # add to graph
        after_filter_graph[axes[ax]] = fx.tolist()
        # Intentionally overwrite... we only care about the last one
        after_filter_size = len(after_filter_graph[axes[ax]])

        if implementation_version >= 3 and analysis_type == 'Wavelet':

            if get_min_length(wavelet_level) > len(fx):
                win_size_ms = np.ceil(get_min_length(wavelet_level) / sampling_freq * 1000).astype('int')
                raise ConfigurationError('Wavelet decomposition level is too high, '
                    f'please increase window size to at least {win_size_ms} ms, or reduce level')

            f, l, approx = dwt_features(fx, wavelet, wavelet_level)
            features.extend(f)
            if ax == 0:
                labels.extend(l)
            wavelet_graph[axes[ax]] = approx.tolist()
            total_sec = len(fx) / sampling_freq
            wavelet_x = np.arange(0, len(approx)) / len(approx) * total_sec
            wavelet_x = (wavelet_x * 1000).astype('int').tolist()

        else:

            if (implementation_version == 1):

                # add root mean square of the features
                features.append(np.sqrt(np.mean(np.square(fx))))
                # add labels as well
                labels = [ 'RMS' ]

                # FFT and frequency domain graph
                fft_res = fft(fx, n=fft_length)
                freq_domain_graph[axes[ax]] = frequency_domain_graph(sampling_freq, fft_res)

                # find spectral peaks
                # returns N peaks with each two values (freq+height) so (N*2) features in total
                px = find_peaks_in_fft(sampling_freq, fft_res, spectral_peaks_threshold, spectral_peaks_count)

                # add them to the features list too
                for peak in px:
                    features.append(peak[0]) # 0 => freq
                    features.append(peak[1]) # 1 => height

                # spectral power edges
                sx = calculate_spectral_power_edges(sampling_freq, fx, spectral_power_edges, n_fft=fft_length)
                # these are N edges each
                for edge in sx:
                    features.append(edge / 10)

                # spectral power graph
                if (draw_graphs):
                    tx, pxx = spectral_power_graph(sampling_freq, fx, n_fft=fft_length)
                    spect_power_graph[axes[ax]] = pxx
                    spect_power_y = tx # all the same so fine here

            else:
                input_decimation_ratio = int(input_decimation_ratio)
                v4 = implementation_version >= 4
                f, labels, spec_powers, freqs = extract_spec_features(
                    fx, sampling_freq, fft_length, filter_type, filter_cutoff,
                    do_log, do_fft_overlap, v4, suffix='')
                features += f

                spect_power_graph[axes[ax]] = spec_powers
                spect_power_y = freqs  # all the same so fine here

                if implementation_version >= 4 and extra_low_freq:
                    fx = decimate_simple(fx, 10)
                    if len(fx) < fft_length / 2:
                        raise ConfigurationError(f'Extra low frequency requires at least {fft_length / 2 * 10} samples')
                    fx = np.array(fx) - np.mean(fx)
                    f, l, _, _= extract_spec_features(
                        fx, sampling_freq / 10, fft_length, filter_type, filter_cutoff,
                        do_log, do_fft_overlap, True, suffix=' LF')
                    lf_features += f
                    if ax == 0:
                        lf_labels = l

    features += lf_features
    labels += lf_labels

    if draw_graphs:
        if butter_sos is not None:
            w, h = signal.sosfreqz(butter_sos, fs=sampling_freq)

            graphs.append({
                'name': 'Filter response',
                'X': [20 * np.log10(zero_handling(abs(h))).tolist()],
                'y': w.tolist(),
                'axisLabels': { 'X': 'Frequency (Hz)', 'y': 'dB' }
            })

        graphs.append({
            'name': 'After filter',
            'X': after_filter_graph,
            'y': np.linspace(0.0, after_filter_size * (1 / sampling_freq) * 1000,
                             num=after_filter_size, endpoint=False).tolist(),
            'suggestedYMin': -1,
            'suggestedYMax': 1,
            'axisLabels': { 'X': 'Sample #', 'y': 'Value' }
        })

        if analysis_type == 'FFT':
            units = 'log' if do_log else 'linear'
            graphs.append({
                'name': f'Spectral power ({units})',
                'X': spect_power_graph,
                'y': spect_power_y,
                'suggestedYMin': 0,
                'suggestedXMin': 0,
                'suggestedXMax': sampling_freq / 2,
                'smoothing': False,
                'axisLabels': { 'X': 'Frequency (Hz)', 'y': 'Energy' }
            })
        elif implementation_version >= 3 and analysis_type == 'Wavelet':

            phi, psi, x = get_wavefunc(wavelet, wavelet_level)
            graphs.append({
                'name': 'Wavelet function',
                'X': [psi.tolist()],
                'y': x.tolist(),
                'axisLabels': { 'X': 'Time', 'y': 'Value' }
            })

            graphs.append({
                'name': 'Wavelet Approximation',
                'X': wavelet_graph,
                'y': wavelet_x,
                'axisLabels': { 'X': 'Time', 'y': 'Value' }
            })

    if (implementation_version == 1):
        for x in range(1, spectral_peaks_count + 1):
            labels.append('Peak ' + str(x) + ' Freq')
            labels.append('Peak ' + str(x) + ' Height')

        for k in range(0, len(spectral_power_edges) - 1):
            labels.append('Spectral Power ' + str(spectral_power_edges[k]) + ' - ' + str(
                spectral_power_edges[k + 1]) + ' Hz')

    return {
        'features': np.array(features).tolist(),
        'graphs': graphs,
        'labels': labels,
        'fft_used': [fft_length],
        'output_config': {'type': 'flat', 'shape': {'width': len(features)}}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Digital Signal Processing script for accelerometer data')
    parser.add_argument('--features', type=str, required=True,
                        help='Axis data as a flattened array of x,y,z (pass as comma separated values)')
    parser.add_argument('--axes', type=str, required=True,
                        help='Names of the axis (pass as comma separated values)')
    parser.add_argument('--frequency', type=float, required=True,
                        help='Frequency in hz')
    parser.add_argument('--scale-axes', type=float, default=1,
                        help='scale axes (divides by this number, default: 1)')
    parser.add_argument('--filter-type', type=str, default='low',
                        help='filter type (low, high or none, default: low)')
    parser.add_argument('--filter-cutoff', type=float, default=3,
                        help='filter cut-off frequency (default: 3)')
    parser.add_argument('--filter-order', type=int, default=6,
                        help='filter order (default: 6)')
    parser.add_argument('--fft-length', type=int, default=128,
                        help='number of FFT points (default: 128)')
    parser.add_argument('--spectral-peaks-count', type=int, default=3,
                        help='number of spectral peaks (default: 3)')
    parser.add_argument('--spectral-peaks-threshold', type=float, default=0.1,
                        help='spectral peaks threshold (default: 0.1)')
    parser.add_argument('--spectral-power-edges', type=str, default='0.1, 0.5, 1.0, 2.0, 5.0',
                        help='spectral power edges (pass as comma separated values, default: "0.1, 0.5, 1.0, 2.0, 5.0")')
    parser.add_argument('--do-log', type=bool, default=False,
                        help='take log of spectrum as features')

    parser.add_argument('--draw-graphs', type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        required=True,
                        help='Whether to draw graphs')

    args = parser.parse_args()

    raw_features = np.array([float(item.strip()) for item in args.features.split(',')])
    raw_axes = args.axes.split(',')

    try:
        processed = generate_features(2, args.draw_graphs, raw_features, raw_axes, args.frequency, args.scale_axes, args.filter_type, args.filter_cutoff,
            args.filter_order, args.fft_length, args.spectral_peaks_count, args.spectral_peaks_threshold, args.spectral_power_edges, args.do_log)

        print('Begin output')
        print(json.dumps(processed))
        print('End output')
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
