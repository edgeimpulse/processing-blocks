import argparse, sys
import json
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis as calculateKurtosis

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes,
                      average, minimum, maximum, rms, stdev, skewness, kurtosis):
    if (implementation_version != 1):
        raise Exception('implementation_version should be 1')

    raw_data = raw_data * scale_axes
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    features = []
    labels = []

    for ax in range(0, len(axes)):
        X = raw_data[:,ax]

        if (average):
            features.append(float(np.average(X)))

        if (minimum):
            features.append(float(np.min(X)))

        if (maximum):
            features.append(float(np.max(X)))

        if (rms):
            features.append(float(np.sqrt(np.mean(np.square(X)))))

        if (stdev):
            features.append(float(np.std(X)))

        if (skewness):
            features.append(float(skew(X)))

        if (kurtosis):
            features.append(float(calculateKurtosis(X)))

    if (average): labels.append('Average')
    if (minimum): labels.append('Minimum')
    if (maximum): labels.append('Maximum')
    if (rms): labels.append('RMS')
    if (stdev): labels.append('Stdev')
    if (skewness): labels.append('Skewness')
    if (kurtosis): labels.append('Kurtosis')

    return {
        'features': features,
        'graphs': [],
        'labels': labels,
        'fft_used': [],
        'output_config': { 'type': 'flat', 'shape': { 'width': len(features) } }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flatten script for raw data')
    parser.add_argument('--features', type=str, required=True,
                        help='Axis data as a flattened array of x,y,z (pass as comma separated values)')
    parser.add_argument('--axes', type=str, required=True,
                        help='Names of the axis (pass as comma separated values)')
    parser.add_argument('--frequency', type=float, required=True,
                        help='Frequency in hz')
    parser.add_argument('--scale-axes', type=float, default=1,
                        help='scale axes (multiplies by this number, default: 1)')
    parser.add_argument('--average', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True,
                        help='calculate average (default: true)')
    parser.add_argument('--minimum', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True,
                        help='calculate minimum (default: true)')
    parser.add_argument('--maximum', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True,
                        help='calculate maximum (default: true)')
    parser.add_argument('--rms', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True,
                        help='calculate rms (default: true)')
    parser.add_argument('--stdev', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True,
                        help='calculate stdev (default: true)')
    parser.add_argument('--skewness', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True,
                        help='calculate skewness (default: true)')
    parser.add_argument('--kurtosis', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True,
                        help='calculate kurtosis (default: true)')
    parser.add_argument('--draw-graphs', type=lambda x: (str(x).lower() in ['true','1', 'yes']), required=True,
                        help='Whether to draw graphs')

    args = parser.parse_args()

    raw_features = np.array([float(item.strip()) for item in args.features.split(',')])
    raw_axes = args.axes.split(',')

    try:
        processed = generate_features(1, args.draw_graphs, raw_features, raw_axes, args.frequency, args.scale_axes,
            args.average, args.minimum, args.maximum, args.rms, args.stdev, args.skewness, args.kurtosis)

        print('Begin output')
        print(json.dumps(processed))
        print('End output')
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
