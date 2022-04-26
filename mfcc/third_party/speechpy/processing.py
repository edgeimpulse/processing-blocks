# -*- coding: utf-8 -*-
"""Processing module for signal processing operations.

This module demonstrates documentation for the signal processing
function which are required as internal computations in the package.


Attributes:

    preemphasis: Preemphasising on the signal. This is a preprocessing step.

    stack_frames: Create stacking frames from the raw signal.

    fft_spectrum: Calculation of the Fast Fourier Transform.

    power_spectrum: Power Spectrum calculation.

    log_power_spectrum: Log Power Spectrum calculation.

    derivative_extraction: Calculation of the derivative of the extracted featurs.

    cmvn: Cepstral mean variance normalization. This is a post processing operation.

    cmvnw: Cepstral mean variance normalization over the sliding window. This is a post processing operation.

"""

__license__ = "MIT"
__author__ = " Amirsina Torfi"
__docformat__ = 'reStructuredText'

import decimal
import numpy as np
import math


# 1.4 becomes 1 and 1.6 becomes 2. special case: 1.5 becomes 2.
def round_half_up(number):
    return int(
        decimal.Decimal(number).quantize(
            decimal.Decimal('1'),
            rounding=decimal.ROUND_HALF_UP))


def preemphasis(signal, shift=1, cof=0.98):
    """preemphasising on the signal.

    Args:
        signal (array): The input signal.
        shift (int): The shift step.
        cof (float): The preemphasising coefficient. 0 equals to no filtering.

    Returns:
           array: The pre-emphasized signal.
    """
    
    if shift <= 0:
        raise ValueError("Shift cannot be zero, or less than 0.  To disable pre emphasis, set coefficient to 0")
    if isinstance(shift, float):
        if not (shift).is_integer():
            raise ValueError("Shift must be a postive integer")

    rolled_signal = np.roll(signal, shift)
    return signal - cof * rolled_signal

def ceil_unless_very_close_to_floor(v):
    """Ceil unless (very) close to floor

     frame_length is a float and can thus be off by a little bit, e.g.
     frame_length = 0.018f actually can yield 0.018000011f
     thus screwing up our frame calculations here...

    Args:
        v: value to ceil or floor
    """
    if (v > np.floor(v)) and (v - np.floor(v) < 0.001):
        v = np.floor(v)
    else:
        v = np.ceil(v)
    return v

def stack_frames(
        sig,
        sampling_frequency,
        implementation_version,
        frame_length=0.020,
        frame_stride=0.020,
        filter=lambda x: np.ones(
            (x,
             )),
        zero_padding=True):
    """Frame a signal into overlapping frames.

    Args:
        sig (array): The audio signal to frame of size (N,).
        sampling_frequency (int): The sampling frequency of the signal.
        frame_length (float): The length of the frame in second.
        frame_stride (float): The stride between frames.
        filter (array): The time-domain filter for applying to each frame.
            By default it is one so nothing will be changed.
        zero_padding (bool): If the samples is not a multiple of
            frame_length(number of frames sample), zero padding will
            be done for generating last frame.

    Returns:
            array: Stacked_frames-Array of frames of size (number_of_frames x frame_len).

    """

    # Check dimension
    s = "Signal dimention should be of the format of (N,) but it is %s instead"
    assert sig.ndim == 1, s % str(sig.shape)

    # Initial necessary values
    length_signal = sig.shape[0]
    if (implementation_version == 1):
        frame_sample_length = int(
            np.round(
                sampling_frequency *
                frame_length))  # Defined by the number of samples
    else:
        frame_sample_length = int(
            ceil_unless_very_close_to_floor(
                sampling_frequency *
                frame_length))  # Defined by the number of samples
    frame_stride = float(ceil_unless_very_close_to_floor(sampling_frequency * frame_stride))

    # Zero padding is done for allocating space for the last frame.
    if zero_padding:
        # Calculation of number of frames
        numframes = (int(math.ceil((length_signal
                                      - frame_sample_length) / frame_stride)))

        # Zero padding
        len_sig = int(numframes * frame_stride + frame_sample_length)
        additive_zeros = np.zeros((len_sig - length_signal,))
        signal = np.concatenate((sig, additive_zeros))

    else:
        # No zero padding! The last frame which does not have enough
        # samples(remaining samples <= frame_sample_length), will be dropped!
        if (implementation_version == 1):
            numframes = int(math.floor((length_signal
                            - frame_sample_length) / frame_stride))
        elif (implementation_version >= 2):
            x = (length_signal - (frame_sample_length - frame_stride))
            numframes = (int(math.floor(x / frame_stride)))
        else:
            raise ValueError('Invalid value for implementation_version, should be 1, 2 or 3 but was ' + (str(implementation_version)))

        # new length
        len_sig = int((numframes - 1) * frame_stride + frame_sample_length)
        signal = sig[0:len_sig]

    # Getting the indices of all frames.
    indices = np.tile(np.arange(0,
                                frame_sample_length),
                      (numframes,
                       1)) + np.tile(np.arange(0,
                                               numframes * frame_stride,
                                               frame_stride),
                                     (frame_sample_length,
                                      1)).T
    indices = np.array(indices, dtype=np.int32)

    # Extracting the frames based on the allocated indices.
    frames = signal[indices]

    # Apply the windows function
    window = np.tile(filter(frame_sample_length), (numframes, 1))
    Extracted_Frames = frames * window
    return Extracted_Frames


def fft_spectrum(frames, fft_points=512):
    """This function computes the one-dimensional n-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT). Please refer to
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html
    for further details.

    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.

    Returns:
            array: The fft spectrum.
            If frames is an num_frames x sample_per_frame matrix, output
            will be num_frames x FFT_LENGTH.
    """
    SPECTRUM_VECTOR = np.fft.rfft(frames, n=fft_points, axis=-1, norm=None)
    return np.absolute(SPECTRUM_VECTOR)


def power_spectrum(frames, fft_points=512):
    """Power spectrum of each frame.

    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.

    Returns:
            array: The power spectrum.
            If frames is an num_frames x sample_per_frame matrix, output
            will be num_frames x fft_length.
    """
    return 1.0 / fft_points * np.square(fft_spectrum(frames, fft_points))


def log_power_spectrum(frames, fft_points=512, normalize=True):
    """Log power spectrum of each frame in frames.

    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than
            frame_len, the frames will be zero-padded.
        normalize (bool): If normalize=True, the log power spectrum
            will be normalized.

    Returns:
           array: The power spectrum - If frames is an
           num_frames x sample_per_frame matrix, output will be
           num_frames x fft_length.
    """
    power_spec = power_spectrum(frames, fft_points)
    power_spec[power_spec <= 1e-20] = 1e-20
    log_power_spec = 10 * np.log10(power_spec)
    if normalize:
        return log_power_spec - np.max(log_power_spec)
    else:
        return log_power_spec


def derivative_extraction(feat, DeltaWindows):
    """This function the derivative features.

    Args:
        feat (array): The main feature vector(For returning the second
             order derivative it can be first-order derivative).
        DeltaWindows (int): The value of  DeltaWindows is set using
            the configuration parameter DELTAWINDOW.

    Returns:
           array: Derivative feature vector - A NUMFRAMESxNUMFEATURES numpy
           array which is the derivative features along the features.
    """

    # Getting the shape of the vector.
    rows, cols = feat.shape

    # Difining the vector of differences.
    DIF = np.zeros(feat.shape, dtype=feat.dtype)
    Scale = 0

    # Pad only along features in the vector.
    FEAT = np.lib.pad(feat, ((0, 0), (DeltaWindows, DeltaWindows)), 'edge')
    for i in range(DeltaWindows):
        # Start index
        offset = DeltaWindows

        # The dynamic range
        Range = i + 1

        dif = Range * FEAT[:, offset + Range:offset + Range + cols]
        - FEAT[:, offset - Range:offset - Range + cols]
        Scale += 2 * np.power(Range, 2)
        DIF += dif

    return DIF / Scale


def cmvn(vec, variance_normalization=False):
    """ This function is aimed to perform global cepstral mean and
        variance normalization (CMVN) on input feature vector "vec".
        The code assumes that there is one observation per row.

    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        variance_normalization (bool): If the variance
            normilization should be performed or not.

    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    eps = 1e-10
    rows, cols = vec.shape

    # Mean calculation
    norm = np.mean(vec, axis=0)
    norm_vec = np.tile(norm, (rows, 1))

    # Mean subtraction
    mean_subtracted = vec - norm_vec

    # Variance normalization
    if variance_normalization:
        stdev = np.std(mean_subtracted, axis=0)
        stdev_vec = np.tile(stdev, (rows, 1))
        output = mean_subtracted / (stdev_vec + eps)
    else:
        output = mean_subtracted

    return output


def cmvnw(vec, win_size=301, variance_normalization=False):
    """ This function is aimed to perform local cepstral mean and
    variance normalization on a sliding window. The code assumes that
    there is one observation per row.

    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        win_size (int): The size of sliding window for local normalization.
            Default=301 which is around 3s if 100 Hz rate is
            considered(== 10ms frame stide)
        variance_normalization (bool): If the variance normilization should
            be performed or not.

    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    # Get the shapes
    eps = 1e-10
    rows, cols = vec.shape

    # Windows size must be odd.
    assert isinstance(win_size, int), "Size must be of type 'int'!"
    assert win_size % 2 == 1, "Windows size must be odd!"

    # Padding and initial definitions
    pad_size = int((win_size - 1) / 2)
    vec_pad = np.lib.pad(vec, ((pad_size, pad_size), (0, 0)), 'symmetric')
    mean_subtracted = np.zeros(np.shape(vec), dtype=np.float32)

    for i in range(rows):
        window = vec_pad[i:i + win_size, :]
        window_mean = np.mean(window, axis=0)
        mean_subtracted[i, :] = vec[i, :] - window_mean

    # Variance normalization
    if variance_normalization:

        # Initial definitions.
        variance_normalized = np.zeros(np.shape(vec), dtype=np.float32)
        vec_pad_variance = np.lib.pad(
            mean_subtracted, ((pad_size, pad_size), (0, 0)), 'symmetric')

        # Looping over all observations.
        for i in range(rows):
            window = vec_pad_variance[i:i + win_size, :]
            window_variance = np.std(window, axis=0)
            variance_normalized[i, :] = mean_subtracted[i, :] / (window_variance + eps)
        output = variance_normalized
    else:
        output = mean_subtracted


    return output


# def resample_Fn(wave, fs, f_new=16000):
#     """This function resample the data to arbitrary frequency
#     :param fs: Frequency of the sound file.
#     :param wave: The sound file itself.
#     :returns:
#            f_new: The new frequency.
#            signal_new: The new signal samples at new frequency.
#
#     dependency: from scikits.samplerate import resample
#     """
#
#     # Resampling using interpolation(There are other
#     methods than 'sinc_best')
#     signal_new = resample(wave, float(f_new) / fs, 'sinc_best')
#
#     # Necessary data converting for saving .wav file using scipy.
#     signal_new = np.asarray(signal_new, dtype=np.int16)
#
#     # # Uncomment if you want to save the audio file
#     # # Save using new format
#     # wav.write(filename='resample_rainbow_16k.wav',rate=fr,data=signal_new)
#     return signal_new, f_new
