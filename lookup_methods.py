import numpy as np
import streamlit as st
import colour
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
        ordered_pre_tone = np.take_along_axis(
            pre_tone, sorted_indices, axis=-1)
        ordered_post_tone = np.take_along_axis(
            post_tone, sorted_indices, axis=-1)

        # linearly interpolate a new post-tone middle channel value between
        # the new (post-tone) max and min channel values.
        high, mid, low = tsplit(ordered_pre_tone)
        new_high, _, new_low = tsplit(ordered_post_tone)

        # compute normalized distance to pre-tone middle channel
        hue_factor = np.where(
            (high - low) != 0., (mid - low) / (high - low), 0.)

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
        ratios = np.ma.divide(RGB, peaks)
        # LUT.apply(
        #     np.clip((2.0**exposure * x), LUT.domain[0], LUT.table[-1])
        # )
        return func(peaks, *args, **kwargs) * ratios.filled(fill_value=0.0)
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

            norm_RGB = nrm_factor * norm(
                np.clip(RGB_out * w, a_min=0, a_max=None))

            # TODO: only apply the weighted lookup to non-negative RGB tuples.
            # TODO: multiply RGB tuples with negative values by the function's
            #       slope at zero

            RGB_out = RGB_out * (func(norm_RGB, *args, **kwargs) / norm_RGB)

            return RGB_out
        return wrapper
    return _decorator(f_py) if callable(f_py) else _decorator


class generic_aesthetic_transfer_function:
    def __init__(
        self,
        contrast=1.0,
        shoulder_contrast=1.0,
        middle_grey_in=0.18,
        middle_grey_out=0.18,
        radiometric_maximum=(2**5)*0.18
    ):
        self.set_transfer_details(
            contrast,
            shoulder_contrast,
            middle_grey_in,
            middle_grey_out,
            radiometric_maximum
        )
        self.calculate_LUT()
        print(self._LUT)

    def set_transfer_details(
        self,
        contrast=1.0,
        shoulder_contrast=1.0,
        middle_grey_in=0.18,
        middle_grey_out=0.18,
        radiometric_maximum=(2**6)*0.18
    ):
        self._contrast = contrast
        self._shoulder_contrast = shoulder_contrast
        self._middle_grey_in = middle_grey_in
        self._middle_grey_out = middle_grey_out
        self._radiometric_maximum = radiometric_maximum

        self._shoulder_multiplied = self._contrast * self._shoulder_contrast

        middle_grey_contrast = \
            pow(self._middle_grey_in, self._contrast)

        middle_grey_shoulder_contrast = \
            pow(self._middle_grey_in, self._shoulder_multiplied)

        radiometric_contrast = \
            pow(self._radiometric_maximum, self._contrast)

        radiometric_multiplied_contrast = \
            pow(self._radiometric_maximum, self._shoulder_multiplied)

        u = radiometric_multiplied_contrast * self._middle_grey_out - \
            middle_grey_shoulder_contrast * self._middle_grey_out
        v = middle_grey_shoulder_contrast * self._middle_grey_out

        self.b = -(
            (
                -middle_grey_contrast + (
                    self._middle_grey_out *
                    (
                        radiometric_multiplied_contrast *
                        middle_grey_contrast - radiometric_contrast * v
                    )
                ) / u
            ) / v
        )
        self.c = (
            radiometric_multiplied_contrast * middle_grey_contrast -
            radiometric_contrast * v
        ) / u

        self.calculate_LUT()

    def evaluate(self, x):
        x = np.minimum(self._radiometric_maximum, x)
        z = np.power(x, self._contrast)
        y = z / (np.power(z, self._shoulder_contrast) * self.b + self.c)

        return np.asarray(y)

    def calculate_LUT(self, LUT_size=1024):
        self._LUT = colour.LUT1D(
            table=self.evaluate(
                np.linspace(0.0, self._radiometric_maximum, LUT_size)),
            name="Generic Lottes 2016 with Fixes",
            domain=[0.0, self._radiometric_maximum + 0.0001]
        )
        self._LUT_size = LUT_size

    def apply(self, RGB, gamut_clip_alert=False):
        gamut_clipped_above = np.where(RGB >= self._radiometric_maximum)

        RGB = np.clip(RGB, 0.0, self._LUT.domain[-1])
        # print(self._radiometric_maximum)
        RGB[gamut_clipped_above[0], gamut_clipped_above[1], :] = \
            self._LUT.domain[-1]
        max_RGBs = np.amax(RGB, axis=2, keepdims=True)
        output_RGBs = self._LUT.apply(max_RGBs) * \
            np.ma.divide(RGB, max_RGBs).filled(fill_value=0.0)
        if gamut_clip_alert is True:
            output_RGBs[gamut_clipped_above[0], gamut_clipped_above[1], :] = \
                [1.0, 0.0, 0.0]
        return output_RGBs
