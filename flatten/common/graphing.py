import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
import math


def set_x_axis_times(frame_stride, frame_length, width):
    plt.xlabel('Time [sec]')
    time_len = (width * frame_stride) + frame_length
    times = np.linspace(0, time_len, 10)
    plt.xticks(np.linspace(0, width, len(times)), [round(x, 2) for x in times])


def create_sgram_graph(sampling_freq, frame_length, frame_stride, width, height, power_spectrum, freqs=None):
    matplotlib.use('Svg')
    _, ax = plt.subplots()
    if not freqs:
        freqs = np.linspace(0, sampling_freq / 2, 15)
    plt.ylabel('Frequency [Hz]')
    ax.imshow(power_spectrum, interpolation='nearest',
              cmap=matplotlib.cm.coolwarm, origin='lower')
    plt.yticks(np.linspace(0, height, len(freqs)), [math.ceil(x) for x in freqs])
    set_x_axis_times(frame_stride, frame_length, width)

    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = (base64.b64encode(buf.getvalue()).decode('ascii'))
    buf.close()
    return image


def create_mfe_graph(sampling_freq, frame_length, frame_stride, width, height, power_spectrum, freqs):
    # Trim down the frequency list for a y axis labels
    freqs = [freqs[0], *freqs[1:-1:4], freqs[-1]]
    return create_sgram_graph(sampling_freq, frame_length, frame_stride, width, height, power_spectrum, freqs)
