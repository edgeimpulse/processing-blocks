import argparse
import json
import math
import numpy as np
import sys
import io, base64
from PIL import Image

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, channels):
    if (implementation_version != 1):
        raise Exception('implementation_version should be 1')

    graphs = []
    pixels = []
    width = raw_data[0]
    height = raw_data[1]
    raw_data = raw_data[2:].astype(dtype=np.uint32).view(dtype=np.uint8)
    bs = raw_data.tobytes()
    ix = 0

    if channels == 'Grayscale':
        while ix < raw_data.shape[0]:
            # ITU-R 601-2 luma transform
            # see: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
            pixels.append((0.299 / 255.0) * float(bs[ix + 2]) + (0.587 / 255.0) * float(bs[ix + 1]) + (0.114 / 255.0) * float(bs[ix]))
            ix = ix + 4
    else:
        while ix < raw_data.shape[0]:
            pixels.append(float(bs[ix + 2]) / 255.0)
            pixels.append(float(bs[ix + 1]) / 255.0)
            pixels.append(float(bs[ix]) / 255.0)
            ix = ix + 4

    if draw_graphs:
        im = None
        if channels == 'Grayscale':
            im = Image.fromarray(np.uint8((np.array(pixels) * 255.0).reshape(height, width)), mode='L')
        else:
            im = Image.fromarray(np.uint8((np.array(pixels) * 255.0).reshape(height, width, 3)), mode='RGB')
        im = im.convert(mode='RGBA')
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

    num_channels = 1
    if channels == 'RGB':
        num_channels = 3

    image_config = { 'width': int(width), 'height': int(height), 'channels': num_channels }
    output_config = { 'type': 'image', 'shape': image_config }

    return { 'features': pixels, 'graphs': graphs, 'output_config': output_config }

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
        processed = generate_features(1, False, raw_features, args.axes, args.frequency, args.channels)

        print('Begin output')
        # print(json.dumps(processed))
        print('End output')
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
