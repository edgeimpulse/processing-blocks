import pywt
import numpy as np
from scipy.stats import skew, entropy, kurtosis


def calculate_entropy(x):
    # todo: try approximate entropy
    # todo: try Kozachenko and Leonenko
    probabilities = np.histogram(x, bins=100, density=True)[0]
    return {'entropy': entropy(probabilities)}


def get_percentile_from_sorted(array, percentile):
    # adding 0.5 is a trick to get rounding out of C flooring behavior during cast
    index = int(((len(array)-1) * percentile/100) + 0.5)
    return array[index]


def calculate_statistics(x):
    output = {}
    x.sort()
    output['n5'] = get_percentile_from_sorted(x, 5)
    output['n25'] = get_percentile_from_sorted(x, 25)
    output['n75'] = get_percentile_from_sorted(x, 75)
    output['n95'] = get_percentile_from_sorted(x, 95)
    output['median'] = get_percentile_from_sorted(x, 50)
    output['mean'] = np.mean(x)
    output['std'] = np.std(x)
    output['var'] = np.var(x, ddof=1)
    output['rms'] = np.sqrt(np.mean(x**2))
    output['skew'] = 0 if output['rms'] == 0 else skew(x)
    output['kurtosis'] = 0 if output['rms'] == 0 else kurtosis(x)
    return output


def calculate_crossings(x):
    lx = len(x)
    zero_crossing_indices = np.nonzero(np.diff(np.array(x) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices) / lx
    m = np.nanmean(x)
    mean_crossing_indices = np.nonzero(np.diff(np.array(x) > m))[0]
    no_mean_crossings = len(mean_crossing_indices) / lx
    return {'zcross': no_zero_crossings, 'mcross': no_mean_crossings}


def get_features(x):
    features = calculate_entropy(x)
    features.update(calculate_crossings(x))
    features.update(calculate_statistics(x))
    return features


def get_max_level(signal_length):
    return int(np.log2(signal_length / 32))


def get_min_length(level):
    return 32 * np.power(2, level)


def dwt_features(x, wav='db4', level=4, mode='stats'):
    y = pywt.wavedec(x, wav, level=level)

    if mode == 'raw':
        XW = [item for sublist in y for item in sublist]
    else:
        features = []
        labels = []
        for i in range(len(y)):
            d = get_features(y[i])
            for k, v in d.items():
                features.append(v)
                labels.append('L' + str(i) + '-' + k)

    return features, labels, y[0]


def get_wavefunc(wav, level):

    wavelet = pywt.Wavelet(wav)
    try:
        phi, psi, x = wavelet.wavefun(level)
    except:
        phi, psi, _, _, x = wavelet.wavefun(level)
    return phi, psi, x
