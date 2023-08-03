import argparse
import json
import numpy as np
import os, sys
from matplotlib import cm
import io, base64
import matplotlib.pyplot as plt
import matplotlib
import math

import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.append(str(ROOT / '..'))
sys.path.append(str(object=ROOT ))
from common.errors import ConfigurationError
from common import graphing

# Load our SpeechPy fork
MODULE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'third_party', 'speechpy', '__init__.py')
MODULE_NAME = 'speechpy'
import importlib
import sys
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
speechpy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = speechpy
spec.loader.exec_module(speechpy)

matplotlib.use('Svg')


def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq,
                      frame_length, frame_stride, num_filters, fft_length,
                      num_cepstral, win_size,
                      low_frequency, high_frequency,
                      pre_cof, pre_shift):
    if (implementation_version > 4):
        raise Exception('implementation_version should be between 1 and 4')

    if (num_filters < 2):
        raise ConfigurationError('Filter number should be at least 2')
    if (not math.log2(fft_length).is_integer()):
        raise ConfigurationError('FFT length must be a power of 2')
    if (len(axes) != 1):
        raise ConfigurationError('MFCC blocks only support a single axis, ' +
            'create one MFCC block per axis under **Create impulse**')

    fs = sampling_freq
    high_frequency = None if high_frequency == 0 else high_frequency

    # reshape first
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    features = []
    graphs = []

    for ax in range(0, len(axes)):
        signal = raw_data[:,ax]

        numframes, _, __ = speechpy.processing.calculate_number_of_frames(
            signal,
            implementation_version=implementation_version,
            sampling_frequency=fs,
            frame_length=frame_length,
            frame_stride=frame_stride,
            zero_padding=False)

        if (numframes < 1):
            raise ConfigurationError('Frame length is larger than your window size')

        if (numframes > 500):
            raise ConfigurationError('Number of frames is larger than 500 (' + str(numframes) + '), ' +
                'increase your frame stride or decrease your window size.')

        # Example of pre-emphasizing.
        signal_preemphasized = speechpy.processing.preemphasis(signal, cof=pre_cof, shift=pre_shift)

        ############# Extract MFCC features #############
        use_old_mels = True if implementation_version <= 3 else False
        mfcc = speechpy.feature.mfcc(signal_preemphasized, sampling_frequency=fs, implementation_version=implementation_version,
                    frame_length=frame_length,
                    frame_stride=frame_stride, num_filters=num_filters, fft_length=fft_length,
                    num_cepstral=num_cepstral,
                    low_frequency=low_frequency, high_frequency=high_frequency,
                    use_old_mels=use_old_mels)
        if win_size > 0:
            mfcc_cmvn = speechpy.processing.cmvnw(mfcc, win_size=win_size, variance_normalization=True)
        else:
            mfcc_cmvn = mfcc

        flattened = mfcc_cmvn.flatten()
        features = np.concatenate((features, flattened))

        if draw_graphs:
            # make visualization too
            fig, ax = plt.subplots()
            fig.set_size_inches(18.5, 20.5)
            # swap to have time on long axis (x)
            # flip so we go from low to high cepstrums
            mfcc_data = np.swapaxes(np.flip(mfcc_cmvn, 1), 0 ,1)
            graphing.set_x_axis_times(frame_stride, frame_length, np.shape(mfcc_data)[1])
            height = np.shape(mfcc_data)[0]
            y_pos = range(0,height)
            y_nums = range(height,0,-1)
            plt.yticks(y_pos,y_nums)
            ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm)

            buf = io.BytesIO()
            plt.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0)

            buf.seek(0)
            image = (base64.b64encode(buf.getvalue()).decode('ascii'))

            buf.close()

            graphs.append({
                'name': 'Cepstral Coefficients',
                'image': image,
                'imageMimeType': 'image/svg+xml',
                'type': 'image'
            })

    return {
        'features': features.tolist(),
        'graphs': graphs,
        'fft_used': [ num_filters, fft_length ],
        'output_config': {
            'type': 'spectrogram',
            'shape': {
                'width': len(features) / num_cepstral,
                'height': num_cepstral
            }
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MFCC script for audio data')
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
    parser.add_argument('--num_filters', type=int, default=32,
                        help='The number of filters in the filterbank')
    parser.add_argument('--fft_length', type=int, default=256,
                        help='Number of FFT points')
    parser.add_argument('--num_cepstral', type=int, default=13,
                        help='Number of Cepstral coefficients')
    parser.add_argument('--win_size', type=int, default=101,
                        help='The size of sliding window for local normalization')
    parser.add_argument('--low_frequency', type=int, default=0,
                        help='Lowest band edge of mel filters')
    parser.add_argument('--high_frequency', type=int, default=0,
                        help='Highest band edge of mel filters. If set to 0 this is equal to samplerate / 2.')
    parser.add_argument('--pre_cof', type=float, default=0.98,
                        help='The preemphasising coefficient. 0 equals to no filtering')
    parser.add_argument('--pre_shift', type=int, default=1,
                        help='')

    args = parser.parse_args()

    raw_features = np.array([float(item.strip()) for item in args.features.split(',')])
    raw_axes = args.axes.split(',')

    try:
        processed = generate_features(2, args.draw_graphs, raw_features, raw_axes, args.frequency,
            args.frame_length, args.frame_stride, args.num_filters, args.fft_length, args.num_cepstral,
            args.win_size, args.low_frequency, args.high_frequency, args.pre_cof, args.pre_shift)

        print('Begin output')
        print(json.dumps(processed))
        print('End output')
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
