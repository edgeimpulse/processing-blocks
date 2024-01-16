import argparse
import json
import numpy as np
import os, sys
import math

import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.append(str(ROOT / '..'))
sys.path.append(str(object=ROOT ))
from common import graphing
from common.errors import ConfigurationError
from third_party import speechpy


def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq,
                      frame_length, frame_stride, fft_length,
                      show_axes, noise_floor_db):
    if (implementation_version >4):
        raise ConfigurationError('implementation_version should be <= 4')
    if (not math.log2(fft_length).is_integer()):
        raise ConfigurationError('FFT length must be a power of 2')
    if (len(axes) != 1):
        raise ConfigurationError('Spectrogram blocks only support a single axis, ' +
            'create one spectrogram block per axis under **Create impulse**')
    if (len(raw_data) < 1):
        raise ConfigurationError('Input data must not be empty')
    if (frame_length < 4/sampling_freq):
        raise ConfigurationError('Frame length should be at least 4 samples')
    if (frame_stride < 4/sampling_freq):
        raise ConfigurationError('Frame stride should be at least 4 samples')

    fs = sampling_freq

    # reshape first
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    features = []
    graphs = []

    width = 0
    height = 0

    for ax in range(0, len(axes)):
        signal = raw_data[:,ax]

        if implementation_version == 3:
            # Rescale to [-1, 1]
            if np.any((signal < -1) | (signal > 1)):
                signal = (signal / 2**15).astype(np.float32)

        sampling_frequency = fs

        s = np.array(signal).astype(float)

        numframes, _, __ = speechpy.processing.calculate_number_of_frames(
            s,
            implementation_version=implementation_version,
            sampling_frequency=sampling_frequency,
            frame_length=frame_length,
            frame_stride=frame_stride,
            zero_padding=False)

        if (numframes < 1):
            raise ConfigurationError('Frame length is larger than your window size')

        if (numframes > 500):
            raise ConfigurationError('Number of frames is larger than 500 (' + str(numframes) + '), ' +
                'increase your frame stride or decrease your window size.')

        frames = speechpy.processing.stack_frames(
            s,
            implementation_version=implementation_version,
            sampling_frequency=sampling_frequency,
            frame_length=frame_length,
            frame_stride=frame_stride,
            filter=lambda x: np.ones((x,)),
            zero_padding=False)

        power_spectrum = speechpy.processing.power_spectrum(frames, fft_length)

        if implementation_version < 3:
            power_spectrum = (power_spectrum - np.min(power_spectrum)) / (np.max(power_spectrum) - np.min(power_spectrum))
            power_spectrum[np.isnan(power_spectrum)] = 0
        else:
            # Clip to avoid zero values
            power_spectrum = np.clip(power_spectrum, 1e-30, None)
            # Convert to dB scale
            # log_mel_spec = 10 * log10(mel_spectrograms)
            power_spectrum = 10 * np.log10(power_spectrum)

            power_spectrum = (power_spectrum - noise_floor_db) / ((-1 * noise_floor_db) + 12)
            max_clip = 1 if implementation_version == 3 else None
            power_spectrum = np.clip(power_spectrum, 0, max_clip)

        flattened = power_spectrum.flatten()
        features = np.concatenate((features, flattened))

        width = np.shape(power_spectrum)[0]
        height = np.shape(power_spectrum)[1]

        if draw_graphs:
            # make visualization too
            power_spectrum = np.swapaxes(power_spectrum, 0, 1)
            image = graphing.create_sgram_graph(sampling_freq, frame_length, frame_stride, width, height, power_spectrum)

            graphs.append({
                'name': 'Spectrogram',
                'image': image,
                'imageMimeType': 'image/svg+xml',
                'type': 'image'
            })

    return {
        'features': features.tolist(),
        'graphs': graphs,
        'fft_used': [ fft_length ],
        'output_config': {
            'type': 'spectrogram',
            'shape': {
                'width': width,
                'height': height
            }
        }
    }





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spectrogram from sensor data')
    parser.add_argument('--features', type=str, required=True,
                        help='Axis data as a flattened WAV file (pass as comma separated values)')
    parser.add_argument('--axes', type=str, required=True,
                        help='Names of the axis (pass as comma separated values)')
    parser.add_argument('--frequency', type=float, required=True,
                        help='Frequency in hz')
    parser.add_argument('--draw-graphs', type=lambda x: (str(x).lower() in ['true','1', 'yes']), required=True,
                        help='Whether to draw graphs')
    parser.add_argument('--frame_length', type=float, default=0.02,
                        help='The length of each frame in seconds')
    parser.add_argument('--frame_stride', type=float, default=0.02,
                        help='The step between successive frames in seconds')
    parser.add_argument('--fft_length', type=int, default=256,
                        help='Number of FFT points')
    parser.add_argument('--noise_floor_db', type=int, default=-52,
                        help='Everything below this loudness will be dropped')
    parser.add_argument('--show-axes', type=lambda x: (str(x).lower() in ['true','1', 'yes']), required=True,
                        help='Whether to show axes on the graph')

    args = parser.parse_args()

    raw_features = np.array([float(item.strip()) for item in args.features.split(',')])
    raw_axes = args.axes.split(',')

    try:
        processed = generate_features(3, args.draw_graphs, raw_features, raw_axes, args.frequency,
            args.frame_length, args.frame_stride, args.fft_length, args.show_axes, args.noise_floor_db)

        print('Begin output')
        print(json.dumps(processed))
        print('End output')
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
