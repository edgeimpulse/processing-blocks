import numpy as np
import sys

sys.path.append('/')
from .errors import ConfigurationError


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def welch_max_hold(fx, sampling_freq, nfft, n_overlap):
    n_overlap = int(n_overlap)
    spec_powers = [0 for _ in range(nfft//2+1)]
    ix = 0
    while ix <= len(fx):
        # Slicing truncates if end_idx > len, and rfft will auto zero pad
        fft_out = np.abs(np.fft.rfft(fx[ix:ix+nfft], nfft))
        spec_powers = np.maximum(spec_powers, fft_out**2/nfft)
        ix = ix + (nfft-n_overlap)
    return np.fft.rfftfreq(nfft, 1/sampling_freq), spec_powers


def zero_handling(x):
    """
    This function handle the issue with zero values if the are exposed
    to become an argument for any log function.
    :param x: The vector.
    :return: The vector with zeros substituted with epsilon values.
    """
    return np.where(x == 0, 1e-10, x)


def cap_frame_stride(window_size_ms, frame_stride):
    """Returns the frame stride passed in,
    or a stride that creates 500 frames if the window size is too large.

    Args:
        window_size_ms (int): The users window size (in ms).
            If none or 0, no capping is done.
        frame_stride (float): The desired frame stride

    Returns:
        float: Either the passed in frame_stride, or longer frame stride
    """
    if window_size_ms:
        num_frames = (window_size_ms / 1000) / frame_stride
        if num_frames > 500:
            print('WARNING: Your window size is too large for the ideal frame stride. '
                f'Set window size to {500 * frame_stride * 1000} ms, or smaller. '
                'Adjusting ideal frame stride to set number of frames to 500')
            frame_stride = (window_size_ms / 1000) / 500
    return frame_stride


def audio_set_params(frame_length, fs):
    """Suggest parameters for audio processing (MFE/MFCC)

    Args:
        frame_length (float): The desired frame length (in seconds)
        fs (int): The sampling frequency (in Hz)

    Returns:
        fft_length: Recomended FFT length
        num_filters: Recomended number of filters
    """
    DEFAULT_NUM_FILTERS = 40
    DEFAULT_NFFT = 256  # for 8kHz sampling rate

    fft_length = next_power_of_2(int(frame_length * fs))
    num_filters = int(DEFAULT_NUM_FILTERS + np.log2(fft_length / DEFAULT_NFFT))
    return fft_length, num_filters
