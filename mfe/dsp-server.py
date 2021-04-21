# This is a generic Edge Impulse DSP server in Python
# You probably don't need to change this file.

import sys, importlib, os, socket, json, math, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
from urllib.parse import urlparse, parse_qs
import traceback
import logging
import numpy as np
from dsp import generate_features

def get_params(self):
    with open('parameters.json', 'r') as f:
        return json.loads(f.read())

def single_req(self, fn, body):
    if (not body['features'] or len(body['features']) == 0):
        raise ValueError('Missing "features" in body')
    if (not 'params' in body):
        raise ValueError('Missing "params" in body')
    if (not 'sampling_freq' in body):
        raise ValueError('Missing "sampling_freq" in body')
    if (not 'draw_graphs' in body):
        raise ValueError('Missing "draw_graphs" in body')

    args = {
        'draw_graphs': body['draw_graphs'],
        'raw_data': np.array(body['features']),
        'axes': np.array(body['axes']),
        'sampling_freq': body['sampling_freq'],
        'implementation_version': body['implementation_version']
    }

    for param_key in body['params'].keys():
        args[param_key] = body['params'][param_key]

    processed = fn(**args)
    if (isinstance(processed['features'], np.ndarray)):
        processed['features'] = processed['features'].tolist()

    body = json.dumps(processed)

    self.send_response(200)
    self.send_header('Content-Type', 'application/json')
    self.end_headers()
    self.wfile.write(body.encode())

def batch_req(self, fn, body):
    if (not body['features'] or len(body['features']) == 0):
        raise ValueError('Missing "features" in body')
    if (not 'params' in body):
        raise ValueError('Missing "params" in body')
    if (not 'sampling_freq' in body):
        raise ValueError('Missing "sampling_freq" in body')

    base_args = {
        'draw_graphs': False,
        'axes': np.array(body['axes']),
        'sampling_freq': body['sampling_freq'],
        'implementation_version': body['implementation_version']
    }

    for param_key in body['params'].keys():
        base_args[param_key] = body['params'][param_key]

    total = 0
    features = []
    labels = []
    output_config = None

    for example in body['features']:
        args = dict(base_args)
        args['raw_data'] = np.array(example)
        f = fn(**args)
        if (isinstance(f['features'], np.ndarray)):
            features.append(f['features'].tolist())
        else:
            features.append(f['features'])

        if total == 0:
            if ('labels' in f):
                labels = f['labels']
            if ('output_config' in f):
                output_config = f['output_config']

        total += 1

    body = json.dumps({
        'success': True,
        'features': features,
        'labels': labels,
        'output_config': output_config
    })

    self.send_response(200)
    self.send_header('Content-Type', 'application/json')
    self.end_headers()
    self.wfile.write(body.encode())

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        params = get_params(self)

        if (url.path == '/'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(('Edge Impulse DSP block: ' + params['info']['title'] + ' by ' +
                params['info']['author']).encode())

        elif (url.path == '/parameters'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            params['version'] = 1
            self.wfile.write(json.dumps(params).encode())

        else:
            self.send_response(404)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Invalid path ' + self.path.encode() + b'\n')

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if (url.path == '/run'):
                content_len = int(self.headers.get('Content-Length'))
                post_body = self.rfile.read(content_len)
                body = json.loads(post_body.decode('utf-8'))
                single_req(self, generate_features, body)

            elif (url.path == '/batch'):
                content_len = int(self.headers.get('Content-Length'))
                post_body = self.rfile.read(content_len)
                body = json.loads(post_body.decode('utf-8'))
                batch_req(self, generate_features, body)

            else:
                self.send_response(404)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Invalid path ' + self.path.encode() + b'\n')


        except Exception as e:
            print('Failed to handle request', e, traceback.format_exc())
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({ 'success': False, 'error': str(e) }).encode())

    def log_message(self, format, *args):
        return

class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    pass

def run():
    host = '0.0.0.0' if not 'HOST' in os.environ else os.environ['HOST']
    port = 4446 if not 'PORT' in os.environ else int(os.environ['PORT'])

    server = ThreadingSimpleServer((host, port), Handler)
    print('Listening on host', host, 'port', port)
    server.serve_forever()

if __name__ == '__main__':
    run()
