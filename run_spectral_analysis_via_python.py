from spectral_analysis import generate_features
import numpy as np

# Parameters for generate_features

## The first section are parameters that apply for any DSP block
# Version of the implementation.  If you want the latest, look into parameters.json, and use the value of latestImplementationVersion
implementation_version = 4 # 4 is latest versions
draw_graphs = False # For testing from script, disable graphing to improve speed

# This is where you want to paste in your test sample.  This can be taken from studio
#  For example, the below sample is from https://studio.edgeimpulse.com/public/223682/latest/classification#load-sample-269491318
#  It was copied by clicking on the copy icon next to "Raw features"
#  It is 3 axis accelerometer data, with 62.5Hz sampling frequency
#  Data should be formatted as a single flat list, regardless of the number of axes/channels
raw_data = np.array([ 2.2600, -1.2700, -1.5300, 1.9500, -1.7500, -1.1900, 1.7900, -2.8500, 0.6500, 1.9100, -2.9100, 2.3500, 1.9100, -2.9100, 2.3500, 1.9900, -2.4100, 3.5900, 1.2700, -0.3800, 2.5200, 1.5300, 0.9900, 2.7300, 2.0200, 0.3000, 4.1400, 1.2400, -0.9500, 5.8300, 0.7400, -1.2500, 6.8400, 0.7400, -1.2500, 6.8400, 0.3100, -0.4200, 6.1200, 1.1300, 0.2500, 8.0100, 1.9600, 0.7400, 10.2900, 1.4600, 0.8700, 9.3800, 0.1200, -0.3400, 9.1400, 0.5500, -1.2900, 12.6500, 0.5500, -1.2900, 12.6500, 1.1000, -1.0500, 14.3300, 0.8300, -0.4800, 12.2600, 0.1900, -0.7900, 11.1900, -0.0500, -0.8100, 12.9700, -0.4500, -0.3500, 17.0500, -0.1000, 0.8300, 17.1800, -0.1000, 0.8300, 17.1800, -0.6000, 0.4200, 14.1100, -0.9000, 0.9200, 12.1500, -0.6000, 1.4200, 14.1400, -0.6200, 1.3100, 15.6600, -0.4800, 1.8700, 14.5600, -0.4300, 1.5300, 13.2100, -0.4300, 1.5300, 13.2100, -0.7800, 1.1400, 12.7500, -0.9100, 1.2100, 13.0900, -0.1600, 1.2200, 14.3900, -0.2900, 1.4000, 13.6000, -1.0200, 1.3800, 13.1400, -1.3400, 0.3400, 14.9300, -1.3400, 0.3400, 14.9300, -1.0000, -1.2300, 19.1200, -1.1200, -3.4000, 19.9800, -1.2800, -2.9000, 19.9800, -1.2400, -1.8300, 19.9800, -1.6200, -2.3900, 19.9800, -1.3900, -1.9000, 19.9800, -1.3900, -1.9000, 19.9800, -1.5300, -1.6200, 19.9800, -1.4700, -0.6200, 19.3400, -1.0400, 1.1900, 17.5700, -1.0400, 1.6700, 15.1700, -1.0600, 1.9200, 12.6000, -0.5100, 3.3900, 12.1800, -0.5100, 3.3900, 12.1800, 0.1600, 3.3500, 14.2200, -0.4200, 1.8600, 13.8700, -0.7000, 1.4700, 10.9600, 0.0000, 2.0200, 8.0200, 0.8800, 2.7000, 6.6000, 1.1700, 2.2400, 7.5300, 1.1700, 2.2400, 7.5300, 0.8600, 0.7700, 9.3000, 1.1300, 0.6500, 9.8600, 0.7000, 0.7600, 6.6400, 0.0000, 0.6500, 2.3700, 0.3300, 0.5600, 0.9900, 0.3300, 0.5600, 0.9900, 1.0200, 0.3000, 1.5100, 0.6200, -0.5900, 1.2600, 0.6800, -1.4500, 1.1200, 0.6900, -2.6900, -0.0900, 0.9100, -2.7700, -1.8600, 1.5000, -2.4300, -2.2100, 1.5000, -2.4300, -2.2100, 2.0400, -3.0300, 0.7100, 2.3600, -2.7900, 3.4400, 2.6200, -2.0500, 3.0800, 2.6000, -1.9000, 0.8300, 2.2700, -2.4700, -0.8600, 2.1600, -3.1600, -1.0300, 2.1600, -3.1600, -1.0300, 2.6200, -2.4300, 0.8500, 3.0000, -1.5500, 2.2800, 2.5900, -0.5700, 2.5000, 1.4900, -0.4300, 1.3700, 0.8600, -0.3800, 2.5600, 1.4900, 0.7300, 4.6100, 1.4900, 0.7300, 4.6100, 1.5300, 1.4200, 4.7200, 1.0800, 1.6100, 4.6600, -0.0700, 1.2600, 5.4800, 0.6200, 1.2700, 8.5500, 1.3700, 0.7600, 11.1200, 1.1400, 0.4600, 10.8000, 1.1400, 0.4600, 10.8000, 0.6400, 1.5400, 10.0600, 0.3400, 0.5300, 10.9900, 0.7600, 1.0000, 14.3200, 0.6400, 2.5200, 15.9600, 0.4300, 2.6400, 17.3400, -0.5500, 1.2500, 13.2400, -0.5500, 1.2500, 13.2400, 0.2300, 2.8200, 14.0200, 0.1300, 3.6500, 15.3500, 0.2300, 3.7600, 16.5700, -0.2900, 2.9500, 15.0900, -0.6000, 3.7800, 15.0900, -0.1100, 4.4300, 15.5600, -0.1100, 4.4300, 15.5600, -0.6600, 3.3100, 16.1400, -0.8000, 2.6700, 16.0700, 0.0900, 3.9500, 16.8200, -0.5600, 4.1900, 16.9800, -0.3700, 3.6100, 19.5200, -0.0300, 1.5200, 19.9800, -0.0300, 1.5200, 19.9800, -0.4200, -0.2900, 19.9800, -0.5100, -0.1200, 19.3700, -0.2500, 0.4800, 19.3000, -0.2500, 0.4300, 19.5900, -0.2700, 0.5000, 19.3400, -0.2700, 0.5000, 17.6800, -0.3400, 1.2800, 17.6800, -0.3200, 2.7600, 15.9600, -0.2200, 3.3800, 14.8100 ])

axes = ['x', 'y', 'z'] # Axes names.  Can be any labels, but the length naturally must match the number of channels in raw data
sampling_freq = 62.5 # Sampling frequency of the data.  Ignored for images

# Below here are parameters that are specific to the spectral analysis DSP block. These are set to the defaults
scale_axes = 1 # Scale your data if desired
input_decimation_ratio = 1 # Decimation ratio.  See /spectral_analysis/paramters.json:31 for valid ratios
filter_type = 'none' # Filter type.  String : low, high, or none
filter_cutoff = 0 # Cutoff frequency if filtering is chosen.  Ignored if filter_type is 'none'
filter_order = 0 # Filter order.  Ignored if filter_type is 'none'.  2, 4, 6, or 8 is valid otherwise
analysis_type = 'FFT' # Analysis type.  String : FFT, wavelet

# The following parameters only apply to FFT analysis type.  Even if you choose wavelet analysis, these parameters still need dummy values
fft_length = 16 # Size of FFT to perform.  Should be power of 2 >-= 16 and <= 4096

# Deprecated parameters.  Only applies to version 1, maintained for backwards compatibility
spectral_peaks_count = 0 # Deprecated parameter.  Only applies to version 1, maintained for backwards compatibility
spectral_peaks_threshold = 0 # Deprecated parameter.  Only applies to version 1, maintained for backwards compatibility
spectral_power_edges = "0" # Deprecated parameter.  Only applies to version 1, maintained for backwards compatibility

# Current FFT parameters
do_log = True # Take the log of the spectral powers from the FFT frames
do_fft_overlap = True # Overlap FFT frames by 50%.  If false, no overlap
extra_low_freq = False # This will decimate the input window by 10 and perform another FFT on the decimated window.
                       # This is useful to extract low frequency data.  The features will be appended to the normal FFT features

# These parameters only apply to Wavelet analysis type.  Even if you choose FFT analysis, these parameters still need dummy values
wavelet_level = 1 # Level of wavelet decomposition
wavelet = "" # Wavelet kernel to use

output = generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes, input_decimation_ratio,
                      filter_type, filter_cutoff, filter_order, analysis_type, fft_length, spectral_peaks_count,
                      spectral_peaks_threshold, spectral_power_edges, do_log, do_fft_overlap,
                      wavelet_level, wavelet, extra_low_freq)

# Return dictionary, as defined in code
    # return {
    #     'features': List of output features
    #     'graphs': Dictionary of graphs
    #     'labels': Names of the features
    #     'fft_used': Array showing which FFT sizes were used.  Helpful for optimzing embedded DSP code
    #     'output_config': information useful for correctly configuring the learn block in Studio
    # }

print(f'Processed features are: ')
print('Feature name, value')
idx = 0
for axis in axes:
    print(f'\nFeatures for axis: {axis}')
    for label in output['labels']:
        print(f'{label: <40}: {output["features"][idx]}')
        idx += 1
