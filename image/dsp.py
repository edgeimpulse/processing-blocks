import argparse
import json
import math
import numpy as np
import sys
import io, base64
from PIL import Image

def generate_features(draw_graphs, raw_data, axes, sampling_freq, channels):
    graphs = []

    raw_data = raw_data.astype(dtype=np.uint32)
    width = raw_data[0]
    height = raw_data[1]

    pixels = []
    for x in np.nditer(raw_data[2:]):
        r = x >> 16 & 0xff
        g = x >> 8 & 0xff
        b = x & 0xff

        pixels.append((b << 16) + (g << 8) + (r))

    im = Image.fromarray(np.array(pixels, dtype=np.uint32).reshape(height, width, 1).view(dtype=np.uint8), mode='RGBA')

    if channels == 'Grayscale':
        im = im.convert(mode='L')
    else:
        im = im.convert(mode='RGB')

    buf = io.BytesIO()
    im.save(buf, format='PNG')

    buf.seek(0)
    image = (base64.b64encode(buf.getvalue()).decode('ascii'))

    buf.close()

    graphs.append({
        'name': 'Image',
        'image': image,
        'imageMimeType': 'image/png',
        'type': 'image'
    })

    features = np.asarray(im) / 255.0
    if channels == 'Grayscale':
        features = features.reshape(width * height)
    else:
        features = features.reshape(width * height * 3)
    return { 'features': features.tolist(), 'graphs': graphs }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Returns raw data')
    parser.add_argument('--features', type=str, required=True,
                        help='Axis data as a flattened WAV file (pass as comma separated values)')
    parser.add_argument('--axes', type=str, required=True,
                        help='Names of the axis (pass as comma separated values)')
    parser.add_argument('--frequency', type=float, required=True,
                        help='Frequency in hz')
    parser.add_argument('--channels', type=str, required=True,
                        help='Image channels to use.')
    parser.add_argument('--draw-graphs', type=bool, required=True,
                        help='Whether to draw graphs')

    args = parser.parse_args()

    raw_features = np.array([float(item.strip()) for item in args.features.split(',')])
    raw_axes = args.axes.split(',')

    try:
        processed = generate_features(False, raw_features, args.axes, args.frequency, args.channels)

        print('Begin output')
        # print(json.dumps(processed))
        print('End output')
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
