import argparse
import json
import numpy as np
import sys

def generate_features(draw_graphs, raw_data, axes, sampling_freq, scale_axes):
    if (scale_axes == 1):
        return { 'features': raw_data, 'graphs': [] }

    return {
        'features': raw_data * scale_axes,
        'graphs': [],
        'output_config': { 'type': 'flat', 'shape': { 'width': len(raw_data) } }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Returns raw data')
    parser.add_argument('--features', type=str, required=True,
                        help='Axis data as a flattened array of x,y,z (pass as comma separated values)')
    parser.add_argument('--axes', type=str, required=True,
                        help='Names of the axis (pass as comma separated values)')
    parser.add_argument('--frequency', type=float, required=True,
                        help='Frequency in hz')
    parser.add_argument('--scale-axes', type=float, default=1,
                        help='scale axes (multiplies by this number, default: 1)')
    parser.add_argument('--draw-graphs', type=bool, required=True,
                        help='Whether to draw graphs')

    args = parser.parse_args()

    raw_features = np.array([float(item.strip()) for item in args.features.split(',')])
    raw_axes = args.axes.split(',')

    try:
        processed = generate_features(args.draw_graphs, raw_features, raw_axes, args.frequency, args.scale_axes)

        print('Begin output')
        print(json.dumps(processed))
        print('End output')
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
