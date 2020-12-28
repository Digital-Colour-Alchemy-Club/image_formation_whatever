import numpy as np
import streamlit as st
from colour.utilities import tsplit, as_float_array
from functools import wraps


def hue_restore_dw3_lookup(func):
    @wraps(func)
    def wrapper(RGB, *args, **kwargs):
        pre_tone = np.ones(3) * np.copy(as_float_array(RGB))
        post_tone = func(pre_tone, *args, **kwargs)
        RGB_out = np.copy(post_tone)

        # pre-tone channel indices, sorted highest to lowest
        sorted_indices = np.argsort(-pre_tone)

        # get index of pre-tone middle channel
        mid_index = sorted_indices[..., 1]

        # re-order pre- and post-tone tuples
        ordered_pre_tone = np.take_along_axis(pre_tone, sorted_indices, axis=-1)
        ordered_post_tone = np.take_along_axis(post_tone, sorted_indices, axis=-1)

        # linearly interpolate a new post-tone middle channel value between
        # the new (post-tone) max and min channel values.
        high, mid, low = tsplit(ordered_pre_tone)
        new_high, _, new_low = tsplit(ordered_post_tone)

        # compute normalized distance to pre-tone middle channel
        hue_factor = np.where((high - low) != 0., (mid - low) / (high - low), 0.)

        # linearly interpolate a new post-tone middle channel value
        new_mid = hue_factor * (new_high - new_low) + new_low

        # update the pre-tone middle channel's post-tone value
        i = np.arange(RGB_out.shape[-2])
        RGB_out[i, mid_index[i]] = new_mid[i]

        return RGB_out


def perchannel_lookup(func):
    @wraps(func)
    def wrapper(RGB, *args, **kwargs):
        return func(RGB)
    return wrapper


def maxrgb_lookup(func):
    @wraps(func)
    def wrapper(RGB, *args, **kwargs):
        peaks = np.amax(RGB, axis=2, keepdims=True)
        ratios = RGB / peaks
        return func(peaks, *args, **kwargs) * ratios
    return wrapper


def norm_lookup(f_py=None, degree=5, weights=(1.22, 1.20, 0.58)):
    # https://stackoverflow.com/a/60832711/13576081
    assert callable(f_py) or f_py is None

    def _decorator(func):
        @wraps(func)
        def wrapper(RGB, *args, **kwargs):
            w = as_float_array(weights)
            RGB_out = np.ones(w.size) * np.copy(as_float_array(RGB))

            def norm(x):
                return np.sum(
                    np.power(x, degree) / np.sum(np.power(x, degree - 1)))

            # compute scaler such that norm(n, n, n) = n
            nrm_factor = 1 / norm(w)

            norm_RGB = nrm_factor * norm(np.clip(RGB_out * w, a_min=0, a_max=None))

            #TODO: only apply the weighted lookup to non-negative RGB tuples.
            #TODO: multiply RGB tuples with negative values by the function's slope at zero

            RGB_out = RGB_out * (func(norm_RGB, *args, **kwargs) / norm_RGB)

            return RGB_out
        return wrapper
    return _decorator(f_py) if callable(f_py) else _decorator
