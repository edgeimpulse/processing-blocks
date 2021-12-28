import argparse
import json, sys, math
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import butter, lfilter
import struct, re, peakutils

def filter(type, freq_hz, cut_off_freq, filt_order, data):
    Wn = cut_off_freq / freq_hz    # Normalized frequency (6 / 62.5) = 0.096

    # Catch when frequency too low
    if (Wn >= 1.0):
        raise Exception('Sample frequency must be greater than ' + str(cut_off_freq) + 'Hz. Try increasing the frequency in the "Impulse design" page.')

    [b, a] = signal.butter(filt_order, Wn=Wn, btype=type)

    filtered_data = lfilter(b,a,data)
    return filtered_data

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
        if (vx[p] < threshold): continue
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

# Unpack a raw sensor data file,
# takes binary file and spits out the sampling frequency, x,y,z vectors with the raw values
# plus a time vector
def unpack_raw_sensor_data_file(data):
    header = data[:data.index(0x0a)].decode('ascii')
    sampling_rate = int(re.match(r'.*?interval (\d+)', header).group(1))
    sampling_freq = 1000 / sampling_rate

    # loop over accel data
    raw_accel = data[data.index(0x0a)+1:]

    x = []
    y = []
    z = []
    time = []

    for ix in range(0, len(raw_accel), 6):
        x.append(struct.unpack('h', raw_accel[ix+0:ix+2])[0])
        y.append(struct.unpack('h', raw_accel[ix+2:ix+4])[0])
        z.append(struct.unpack('h', raw_accel[ix+4:ix+6])[0])
        time.append((ix / 6) * sampling_rate)

    x = list(map(lambda v: v / 100, x))
    y = list(map(lambda v: v / 100, y))
    z = list(map(lambda v: v / 100, z))

    return sampling_freq, x, y, z, time

# You can test this function out visually in visualizer.py
def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes,
                      filter_type, filter_cutoff, filter_order, fft_length, spectral_peaks_count,
                      spectral_peaks_threshold, spectral_power_edges):
    if (implementation_version != 1):
        raise Exception('implementation_version should be 1')

    if (filter_type == 'low' or filter_type == 'high'):
        if (filter_order % 2 != 0):
            raise Exception('Filter order needs to be even (2, 4, 6, 8)')
        if (filter_order < 1 or filter_order > 9):
            raise Exception('Filter order needs to be between 2 and 8')

    if (not math.log2(fft_length).is_integer()):
        raise Exception('FFT length must be a power of 2')

    # reshape first
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    if (isinstance(spectral_power_edges, str)):
        spectral_power_edges = [float(item.strip()) for item in spectral_power_edges.split(',')]

    features = []
    graphs = []

    after_filter_graph = {}
    freq_domain_graph = {}
    spect_power_graph = {}
    spect_power_y = []

    for ax in range(0, len(axes)):
        X = raw_data[:,ax]

        # potentially scale data from sensor
        fx = np.array(X, dtype='f') * scale_axes

        # offset by the mean
        fx = np.array(fx) - np.mean(fx)

        # apply lowpass filter to denoise
        if (filter_type == 'low' or filter_type == 'high'):
            fx = filter(filter_type, sampling_freq, filter_cutoff * 2, filter_order, fx)

        # add to graph
        after_filter_graph[axes[ax]] = fx.tolist()

        # add root mean square of the features
        features.append(np.sqrt(np.mean(np.square(fx))))

        # find spectral peaks
        # returns 3 peaks with each two values (freq+height) so 6 features in total
        px = find_peaks_in_fft(sampling_freq, fft(fx, n=fft_length), spectral_peaks_threshold, spectral_peaks_count)

        freq_domain_graph[axes[ax]] = frequency_domain_graph(sampling_freq, fft(fx, n=fft_length))

        # add them to the features list too
        for peak in px:
            features.append(peak[0]) # 0 => freq
            features.append(peak[1]) # 1 => height

        # spectral power edges
        sx = calculate_spectral_power_edges(sampling_freq, fx, spectral_power_edges, n_fft=fft_length)
        # these are 4 edges each
        for edge in sx:
            features.append(edge / 10)

        # spectral power graph
        tx, pxx = spectral_power_graph(sampling_freq, fx, n_fft=fft_length)
        spect_power_graph[axes[ax]] = pxx
        spect_power_y = tx # all the same so fine here

    graphs.append({
        'name': 'After filter',
        'X': after_filter_graph,
        'y': np.linspace(0.0, len(fx) * (1 / sampling_freq) * 1000, len(fx) + 1).tolist(),
        'suggestedYMin': -1,
        'suggestedYMax': 1
    })

    graphs.append({
        'name': 'Frequency domain',
        'X': freq_domain_graph,
        'y': frequency_domain_graph_y(sampling_freq, raw_data.shape[0]),
        'suggestedYMin': 0,
        'suggestedYMax': 2,
        'suggestedXMin': 0,
        'suggestedXMax': sampling_freq / 2
    })

    graphs.append({
        'name': 'Spectral power',
        'X': spect_power_graph,
        'y': spect_power_y,
        'suggestedXMin': 0,
        'suggestedXMax': sampling_freq / 2,
        'suggestedYMin': 1e-7,
        'type': 'logarithmic'
    })

    # add labels as well
    labels = [
        'RMS'
    ]

    for x in range(1, spectral_peaks_count + 1):
        labels.append('Peak ' + str(x) + ' Freq')
        labels.append('Peak ' + str(x) + ' Height')

    for k in range(0, len(spectral_power_edges) - 1):
        labels.append('Spectral Power ' + str(spectral_power_edges[k]) + ' - ' + str(spectral_power_edges[k + 1]))

    return {
        'features': np.array(features).tolist(),
        'graphs': graphs,
        'labels': labels,
        'fft_used': [ fft_length ],
        'output_config': { 'type': 'flat', 'shape': { 'width': len(features) } }
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
    parser.add_argument('--draw-graphs', type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        required=True,
                        help='Whether to draw graphs')

    args = parser.parse_args()

    raw_features = np.array([float(item.strip()) for item in args.features.split(',')])
    raw_axes = args.axes.split(',')

    try:
        processed = generate_features(1, args.draw_graphs, raw_features, raw_axes, args.frequency, args.scale_axes, args.filter_type, args.filter_cutoff,
            args.filter_order, args.fft_length, args.spectral_peaks_count, args.spectral_peaks_threshold, args.spectral_power_edges)

        print('Begin output')
        print(json.dumps(processed))
        print('End output')
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
