import argparse
import json
import numpy as np
import os, sys
from matplotlib import cm
import io, base64
import matplotlib.pyplot as plt
import time
import matplotlib
from scipy import signal as sn
import math

# Load our SpeechPy fork
MODULE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third_party', 'speechpy', '__init__.py')
MODULE_NAME = 'speechpy'
import importlib
import sys
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
import speechpy

matplotlib.use('Svg')

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq,
                      frame_length, frame_stride, fft_length,
                      show_axes, noise_floor_db):
    if (implementation_version != 1 and implementation_version != 2 and implementation_version != 3):
        raise Exception('implementation_version should be 1, 2 or 3')

    fs = sampling_freq

    # reshape first
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    features = []
    graphs = []

    width = 0
    height = 0

    for ax in range(0, len(axes)):
        signal = raw_data[:,ax]

        if implementation_version >= 3:
            # Rescale to [-1, 1] and add preemphasis
            signal = (signal / 2**15).astype(np.float32)

        sampling_frequency = fs

        s = np.array(signal).astype(float)
        frames = speechpy.processing.stack_frames(
            s,
            implementation_version=implementation_version,
            sampling_frequency=sampling_frequency,
            frame_length=frame_length,
            frame_stride=frame_stride,
            filter=lambda x: np.ones(
                (x,
                    )),
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
            power_spectrum = np.clip(power_spectrum, 0, 1)

        flattened = power_spectrum.flatten()
        features = np.concatenate((features, flattened))

        width = np.shape(power_spectrum)[0]
        height = np.shape(power_spectrum)[1]

        if draw_graphs:
            # make visualization too
            power_spectrum = np.swapaxes(power_spectrum, 0, 1)
            fig, ax = plt.subplots()

            if not show_axes:
                cax = ax.imshow(power_spectrum, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
                fig.set_size_inches(18.5, 20.5)
                ax.set_axis_off()
            else:
                time_len = (width * frame_stride) + frame_length
                times = np.linspace(0, time_len, 10)
                freqs = np.linspace(0, sampling_freq / 2, 15)
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                cax = ax.imshow(power_spectrum, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
                plt.xticks(np.linspace(0, width, 10), [ round(x, 2) for x in times ])
                plt.yticks(np.linspace(0, height, 15), [ math.ceil(x) for x in freqs ])

            buf = io.BytesIO()

            plt.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0)

            buf.seek(0)
            image = (base64.b64encode(buf.getvalue()).decode('ascii'))

            buf.close()

            graphs.append({
                'name': 'Spectrogram',
                'image': image,
                'imageMimeType': 'image/svg+xml',
                'type': 'image'
            })

    return {
        'features': features.tolist(),
        'graphs': graphs,
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
        processed = generate_features(2, args.draw_graphs, raw_features, raw_axes, args.frequency,
            args.frame_length, args.frame_stride, args.num_filters, args.fft_length, args.noise_floor_db)

        print('Begin output')
        print(json.dumps(processed))
        print('End output')
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
