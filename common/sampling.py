import math


def calc_resampled_size(input_sample_rate, output_sample_rate, input_length):
    """Calculate the output size after resampling.
    :returns: integer output size, >= 1
    """
    target_size = int(math.ceil((output_sample_rate / input_sample_rate) * (input_length)))
    return max(target_size, 1)
