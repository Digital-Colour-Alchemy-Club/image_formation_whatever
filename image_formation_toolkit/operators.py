from functools import partial
from typing import List, Optional
from six import string_types
import attr
from boltons.strutils import camel2under, under2camel, slugify
from colour.io import AbstractLUTSequenceOperator, LUT3D, LUT1D, LUTSequence
from colour.models.rgb.transfer_functions.log import *
from colour.models.rgb.transfer_functions.exponent import *
from colour.utilities import as_float_array, tstack, tsplit, filter_kwargs
from colour.algebra.common import spow
import numpy as np


# from colour
def vector_dot(m, v):
    """
    Convenient wrapper around :func:`np.einsum` with the following subscripts:
    *'...ij,...j->...i'*.

    It performs the dot product of two arrays where *m* parameter is expected
    to be an array of 3x3 matrices and parameter *v* an array of vectors.

    Parameters
    ----------
    m : array_like
        Array of 3x3 matrices.
    v : array_like
        Array of vectors.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> m = np.array(
    ...     [[0.7328, 0.4296, -0.1624],
    ...      [-0.7036, 1.6975, 0.0061],
    ...      [0.0030, 0.0136, 0.9834]]
    ... )
    >>> m = np.reshape(np.tile(m, (6, 1)), (6, 3, 3))
    >>> v = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> v = np.tile(v, (6, 1))
    >>> vector_dot(m, v)  # doctest: +ELLIPSIS
    array([[ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...]])
    """

    m = as_float_array(m)
    v = as_float_array(v)

    return np.einsum("...ij,...j->...i", m, v)


@attr.s(auto_attribs=True, frozen=True)
class AestheticTransferFunction(AbstractLUTSequenceOperator):
    name: str = "Generic Lottes (2016) with Fixes"
    comments: Optional[List[string_types]] = None
    contrast: float = 1.0
    shoulder_contrast: float = 1.0
    middle_grey_in: float = 0.18
    middle_grey_out: float = 0.18
    ev_above_middle_grey: float = attr.ib(default=4.0)
    ev_below_middle_grey: float = attr.ib(
        default=-6.5, converter=lambda x: -abs(float(x))
    )
    per_channel_lookup: bool = False
    gamut_clip: bool = True
    gamut_warning: bool = False

    @ev_above_middle_grey.validator
    def _check_ev_above_middle_grey(self, attribute, value):
        return 1.0 <= value <= 20.0

    @property
    def radiometric_maximum(self):
        ev = np.clip(self.ev_above_middle_grey, 1.0, 20.0)
        return spow(2.0, ev) * self.middle_grey_in

    @property
    def radiometric_minimum(self):
        ev = np.clip(self.ev_below_middle_grey, -20.0, -1.0)
        return spow(2.0, ev) * self.middle_grey_in

    def _apply(self, RGB, preserve_negatives=False):
        RGB_out = np.copy(as_float_array(RGB))

        shoulder_multiplied = self.contrast * self.shoulder_contrast

        middle_grey_contrast = spow(self.middle_grey_in, self.contrast)

        middle_gray_shoulder_contrast = spow(self.middle_grey_in, shoulder_multiplied)

        radiometric_contrast = spow(self.radiometric_maximum, self.contrast)

        radiometric_multiplied_contrast = spow(
            self.radiometric_maximum, shoulder_multiplied
        )

        v = middle_gray_shoulder_contrast * self.middle_grey_out
        u = radiometric_multiplied_contrast * self.middle_grey_out - v
        a = (
            radiometric_multiplied_contrast * middle_grey_contrast
            - radiometric_contrast * v
        )
        b = -((-middle_grey_contrast + (self.middle_grey_out * a) / u) / v)
        c = a / u

        negative_values = np.where(RGB_out < 0.0)
        if preserve_negatives:
            RGB_out = np.abs(RGB_out)

        RGB_out = np.minimum(self.radiometric_maximum, RGB_out)
        RGB_out = spow(RGB_out, self.contrast)
        RGB_out = RGB_out / (spow(RGB_out, self.shoulder_contrast) * b + c)

        if preserve_negatives:
            RGB_out[negative_values] = np.negative(RGB_out)

        return RGB_out

    def _apply_shaped(self, shaped_RGB):
        min_exposure = self.ev_below_middle_grey
        unshaped_RGB = log_decoding_Log2(
            shaped_RGB,
            middle_grey=self.middle_grey_in,
            min_exposure=min_exposure,
            max_exposure=self.ev_above_middle_grey,
        )
        return self.apply(unshaped_RGB)

    def apply(self, RGB, *args):
        RGB_out = np.copy(as_float_array(RGB))
        gamut_clipped_above = np.where(RGB_out >= self.radiometric_maximum)
        gamut_clipped_below = np.where(RGB_out < self.radiometric_minimum)

        if self.per_channel_lookup:
            RGB_out = self._apply(RGB_out)
        else:
            RGB_max = np.amax(RGB_out, axis=-1, keepdims=True)
            ratios = np.ma.divide(RGB_out, RGB_max).filled(fill_value=0.0)
            RGB_out = self._apply(RGB_max) * ratios

        if self.gamut_clip:
            max_val = self._apply(self.radiometric_maximum)
            min_val = self._apply(self.radiometric_minimum)
            RGB_out[gamut_clipped_above[0], gamut_clipped_above[1], :] = max_val
            RGB_out[gamut_clipped_below] = min_val

        if self.gamut_warning and (RGB_out.size > 1 or RGB_out.shape[-1] > 1):
            warning = np.array([1.0, 0.0, 0.0])
            RGB_out[gamut_clipped_above[0], gamut_clipped_above[1], :] = warning

        return RGB_out

    def generate_lut1d3d(self, size=33, shaper_size=2**14):
        shaper_to_lin = partial(
            log_decoding_Log2,
            middle_grey=self.middle_grey_in,
            min_exposure=self.ev_below_middle_grey,
            max_exposure=self.ev_above_middle_grey,
        )
        lin_to_shaper = partial(
            log_encoding_Log2,
            middle_grey=self.middle_grey_in,
            min_exposure=self.ev_below_middle_grey,
            max_exposure=self.ev_above_middle_grey,
        )
        shaper = LUT1D(
            size=shaper_size,
            name=f"{self.name} -- Shaper",
            domain=shaper_to_lin([0.0, 1.0]),
            comments=[
                f"{'Min Exposure:':<20}{self.ev_below_middle_grey}",
                f"{'Max Exposure:':<20}{self.ev_above_middle_grey}",
                f"{'Middle Grey:':<20}{self.middle_grey_in}",
            ],
        )
        shaper.table = lin_to_shaper(shaper.table)
        cube = LUT3D(
            size=size,
            name=self.name,
            comments=self.comments,
        )
        cube.table = self._apply_shaped(cube.table)
        return LUTSequence(shaper, cube)

    def generate_clf(self, size=33):
        cube = LUT3D(size=size, name=self.name, comments=self.comments)
        radiometric_minimum = 0.18 * spow(2, self.ev_below_middle_grey)
        shaper = Range(
            min_in_value=radiometric_minimum,
            max_in_value=self.radiometric_maximum,
            min_out_value=0,
            max_out_value=1,
            no_clamp=not self.gamut_clip,
            name=f"{self.name} -- shaper",
        )
        inv_shaper = Range(
            min_in_value=0,
            max_in_value=1,
            min_out_value=radiometric_minimum,
            max_out_value=self.radiometric_maximum,
            no_clamp=not self.gamut_clip,
        )

        cube.table = inv_shaper.apply(cube.table)
        cube.table = self.apply(cube.table)
        return LUTSequence(shaper, cube)

    def __str__(self):
        return (
            "{0} - {1}\n"
            "{2}\n\n"
            f"{'Contrast:':<20}{self.contrast}\n"
            f"{'Shoulder Contrast:':<20}{self.shoulder_contrast}\n"
            f"{'Mid Grey In:':<20}{self.middle_grey_in}\n"
            f"{'Mid Grey Out:':<20}{self.middle_grey_out}\n"
            f"{'EV Above Mid Grey:':<20}{np.clip(self.ev_above_middle_grey, 1., 20.)}\n"
            f"{'MaxRGB:':<20}{'ON' if not self.per_channel_lookup else 'OFF'}\n"
            f"{'Gamut Clip:':<20}{'ON' if self.gamut_clip else 'OFF'}\n"
            f"{'Gamut Warning:':<20}{'ON' if self.gamut_warning else 'OFF'}\n"
            "{3}"
        ).format(
            self.__class__.__name__,
            self.name,
            "-" * (len(self.__class__.__name__) + 3 + len(self.name)),
            "\n\n{0}".format("\n".join(self.comments)) if self.comments else "",
        )


# From Nick Shaw's colour-science feature/CLF branch
class Range(AbstractLUTSequenceOperator):
    """
    Defines the class for a *Range* scale.

    Parameters
    ----------
    min_in_value : numeric, optional
        Input value which will be mapped to min_out_value.
    max_in_value : numeric, optional
        Input value which will be mapped to max_out_value.
    min_out_value : numeric, optional
        Output value to which min_in_value will be mapped.
    max_out_value : numeric, optional
        Output value to which max_in_value will be mapped.
    no_clamp : boolean, optional
        Whether to not clamp the output values.
    name : unicode, optional
        *Range* name.
    comments : array_like, optional
        Comments to add to the *Range*.

    Methods
    -------
    apply

    Examples
    --------
    A full to legal scale:

    >>> print(Range(name='Full to Legal',
                    min_out_value=64./1023,
                    max_out_value=940./1023))
    Range - Full to Legal
    ---------------------
    <BLANKLINE>
    Input      : 0.0 - 1.0
    Output     : 0.0625610948192 - 0.918866080156
    <BLANKLINE>
    Clamping   : No
    """

    def __init__(
        self,
        min_in_value=0.0,
        max_in_value=1.0,
        min_out_value=0.0,
        max_out_value=1.0,
        no_clamp=True,
        name="",
        comments=None,
    ):
        self.min_in_value = min_in_value
        self.max_in_value = max_in_value
        self.min_out_value = min_out_value
        self.max_out_value = max_out_value
        self.no_clamp = no_clamp
        self.name = name
        self.comments = comments

    def apply(self, RGB):
        """
        Applies the *Range* scale to given *RGB* array.

        Parameters
        ----------
        RGB : array_like
            *RGB* array to apply the *Range* scale to.

        Returns
        -------
        ndarray
            Scaled *RGB* array.

        Examples
        --------
        >>> R = Range(name='Legal to Full',
        ...           min_in_value=64./1023,
        ...           max_in_value=940./1023,
        ...           no_clamp=False)
        >>> RGB = np.array([0.8, 0.9, 1.0])
        >>> R.apply(RGB)
        array([ 0.86118721,  0.97796804,  1.        ])
        """
        RGB = np.asarray(RGB)

        scale = (self.max_out_value - self.min_out_value) / (
            self.max_in_value - self.min_in_value
        )
        RGB_out = RGB * scale + self.min_out_value - self.min_in_value * scale

        if not self.no_clamp:
            RGB_out = np.clip(RGB_out, self.min_out_value, self.max_out_value)

        return RGB_out

    @classmethod
    def from_ocio(cls, transform):
        min_in, max_in, min_out, max_out = (
            transform.getMinInValue(),
            transform.getMaxInValue(),
            transform.getMinOutValue(),
            transform.getMaxOutValue(),
        )
        if transform.getDirection() == 1:
            max_in, max_out = max_out, max_in
            min_in, min_out = min_out, min_in
        return cls(
            min_in_value=min_in,
            max_in_value=max_in,
            min_out_value=min_out,
            max_out_value=max_out,
            no_clamp=transform.getStyle() == 0,
        )

    @classmethod
    def from_camel_case_dict(cls, data):
        return cls(**filter_kwargs(cls, **{camel2under(k): v for k, v in data.items()}))

    def to_camel_case_dict(self):
        # todo: lower camel case...
        return {under2camel(k): v for k, v in self.__dict__.items()}

    def __str__(self):
        """
        Returns a formatted string representation of the *Range* operation.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return (
            "{0} - {1}\n"
            "{2}\n\n"
            "Input      : {3} - {4}\n"
            "Output     : {5} - {6}\n\n"
            "Clamping   : {7}"
            "{8}".format(
                self.__class__.__name__,
                self.name,
                "-" * (len(self.__class__.__name__) + 3 + len(self.name)),
                self.min_in_value,
                self.max_in_value,
                self.min_out_value,
                self.max_out_value,
                "No" if self.no_clamp else "Yes",
                "\n\n{0}".format("\n".join(self.comments)) if self.comments else "",
            )
        )


# From Nick Shaw's colour-science feature/CLF branch
class Matrix(AbstractLUTSequenceOperator):
    """
    Defines the base class for a *Matrix* transform.

    Parameters
    ----------
    array : array_like, optional
        3x3 or 3x4 matrix for the transform.
    name : unicode, optional
        *Matrix* name.
    comments : array_like, optional
        Comments to add to the *Matrix*.

    Methods
    -------
    apply

    Examples
    --------
    Instantiating an identity matrix:

    >>> print(Matrix(name='Identity'))
    Matrix - Identity
    -----------------
    <BLANKLINE>
    Dimensions : (3, 3)
    Matrix     : [[ 1.  0.  0.]
                  [ 0.  1.  0.]
                  [ 0.  0.  1.]]

    Instantiating a matrix with comments:

    >>> array = np.array([[ 1.45143932, -0.23651075, -0.21492857],
        ...                   [-0.07655377,  1.1762297 , -0.09967593],
        ...                   [ 0.00831615, -0.00603245,  0.9977163 ]])
    >>> print(Matrix(array=array,
    ...       name='AP0 to AP1',
    ...       comments=['A first comment.', 'A second comment.']))
    Matrix - AP0 to AP1
    -------------------
    <BLANKLINE>
    Dimensions : (3, 3)
    Matrix     : [[ 1.45143932 -0.23651075 -0.21492857]
                  [-0.07655377  1.1762297  -0.09967593]
                  [ 0.00831615 -0.00603245  0.9977163 ]]
    <BLANKLINE>
    A first comment.
    A second comment.
    """

    def __init__(self, array=np.identity(3), name="", comments=None):
        self.array = array
        self.name = name
        self.comments = comments

    @staticmethod
    def _validate_array(array):
        assert array.shape in [(3, 4), (3, 3)], "Matrix shape error!"

        return array

    def apply(self, RGB):
        """
        Applies the *Matrix* transform to given *RGB* array.

        Parameters
        ----------
        RGB : array_like
            *RGB* array to apply the *Matrix* transform to.

        Returns
        -------
        ndarray
            Transformed *RGB* array.

        Examples
        --------
        >>> array = np.array([[ 1.45143932, -0.23651075, -0.21492857],
        ...                   [-0.07655377,  1.1762297 , -0.09967593],
        ...                   [ 0.00831615, -0.00603245,  0.9977163 ]])
        >>> M = Matrix(array=array)
        >>> RGB = [0.3, 0.4, 0.5]
        >>> M.apply(RGB)
        array([ 0.23336321,  0.39768778,  0.49894002])
        """
        RGB = np.asarray(RGB)

        if self.array.shape == (3, 4):
            R, G, B = tsplit(RGB)
            RGB = tstack([R, G, B, np.ones(R.shape)])

        return vector_dot(self.array, RGB)

    def __str__(self):
        """
        Returns a formatted string representation of the *Matrix*.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        def _indent_array(a):
            """
            Indents given array string representation.
            """

            return str(a).replace(" [", " " * 14 + "[")

        return (
            "{0} - {1}\n"
            "{2}\n\n"
            "Dimensions : {3}\n"
            "Matrix     : {4}"
            "{5}".format(
                self.__class__.__name__,
                self.name,
                "-" * (len(self.__class__.__name__) + 3 + len(self.name)),
                self.array.shape,
                _indent_array(self.array),
                "\n\n{0}".format("\n".join(self.comments)) if self.comments else "",
            )
        )


# From Nick Shaw's colour-science feature/CLF branch
class Exponent(AbstractLUTSequenceOperator):
    def __init__(
        self,
        exponent=[1, 1, 1],
        offset=[0, 0, 0],  # ignored for basic
        style="basicFwd",
        name="",
        comments=None,
    ):
        self.exponent = exponent
        self.offset = offset
        self.style = style
        self.name = name
        self.comments = comments

    def apply(self, RGB):
        if as_float_array(RGB).size == 3 or (
            isinstance(RGB, np.ndarray) and RGB.shape[-1] == 3
        ):
            r, g, b = tsplit(np.asarray(RGB))

        else:
            r = g = b = np.asarray(RGB)

        if self.style.lower()[:5] == "basic":
            r = exponent_function_basic(r, self.exponent[0], self.style)
            g = exponent_function_basic(g, self.exponent[1], self.style)
            b = exponent_function_basic(b, self.exponent[2], self.style)

            return tstack((r, g, b))

        if self.style.lower()[:8] == "moncurve":
            r = exponent_function_monitor_curve(
                r, self.exponent[0], self.offset[0], self.style
            )
            g = exponent_function_monitor_curve(
                g, self.exponent[1], self.offset[1], self.style
            )
            b = exponent_function_monitor_curve(
                b, self.exponent[2], self.offset[2], self.style
            )

            return tstack((r, g, b))

    def __str__(self):
        return (
            "{0} - {1}\n"
            "{2}\n\n"
            "Exponent.r : {3}\n"
            "Exponent.g : {4}\n"
            "Exponent.b : {5}\n"
            "{6}"
            "Style : {7}\n"
            "{8}".format(
                self.__class__.__name__,
                self.name,
                "-" * (len(self.__class__.__name__) + 3 + len(self.name)),
                self.exponent[0],
                self.exponent[1],
                self.exponent[2],
                (
                    "Offset.r : {}\nOffset.g : {}\nOffset.b : {}\n".format(
                        self.offset[0], self.offset[1], self.offset[2]
                    )
                    if self.style.lower()[:8] == "moncurve"
                    else ""
                ),
                self.style,
                "\n\n{0}".format("\n".join(self.comments)) if self.comments else "",
            )
        )


class Log(AbstractLUTSequenceOperator):
    # TODO: Actual docstrings :)
    """
    >>> from colour.models.rgb.transfer_functions.arri_alexa_log_c import (
    ...      log_decoding_ALEXALogC, log_encoding_ALEXALogC)
    >>> logc = Log(logSideSlope=0.24719, logSideOffset=0.385537,
    ...            linSideSlope=5.55556,  linSideOffset=0.0522723,
    ...            linSideBreak=0.010591)
    ... x = np.linspace(-0.33, 1.33, 1024)
    ... RGB = tstack([x, x, x])
    ... assert np.allclose(log_encoding_ALEXALogC(x), logc.apply(x))
    ... assert np.allclose(log_decoding_ALEXALogC(RGB), logc.reverse(RGB))
    ... assert np.array_equal(RGB, logc.apply(logc.reverse(RGB)))
    ... assert np.array_equal(RGB, logc.reverse(logc.apply(RGB)))
    """

    def __init__(
        self,
        linSideSlope=1,
        linSideOffset=0,
        logSideSlope=1,
        logSideOffset=0,
        linSideBreak=None,
        linearSlope=None,
        base=10,
        style="linToLog",
        name="",
        comments=None,
    ):
        self._base = None
        self.base = base
        self.name = name
        self.style = style
        self.comments = comments or []

        self.linSideOffset = linSideOffset
        self.linSideSlope = linSideSlope
        self.logSideSlope = logSideSlope
        self.logSideOffset = logSideOffset
        self.linSideBreak = linSideBreak
        self.linearSlope = linearSlope

    @property
    def lin_to_log_styles(self):
        return ["log2", "log10", "linToLog", "cameraLinToLog"]

    @property
    def log_to_lin_styles(self):
        return ["antiLog2", "antiLog10", "logToLin", "cameraLogToLin"]

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, value):
        self._base = int(value) if float(value) == int(value) else float(value)

    @property
    def style(self):
        style = self._style
        if style.startswith("camera") and self.lin_side_break is None:
            style = style.replace("cameraL", "l")
        return style

    @style.setter
    def style(self, value):
        if not value in self.log_styles:
            raise ValueError("Invalid Log style: %s" % value)

        if value.endswith("2"):
            self.base = 2
        elif value.endswith("10"):
            self.base = 10
        else:
            self._style = value

    @property
    def log_styles(self):
        return self.log_to_lin_styles + self.lin_to_log_styles

    @property
    def log_side_slope(self):
        return self.logSideSlope

    @log_side_slope.setter
    def log_side_slope(self, value):
        self.logSideSlope = value

    @property
    def log_side_offset(self):
        return self.logSideOffset

    @log_side_offset.setter
    def log_side_offset(self, value):
        self.logSideOffset = value

    @property
    def lin_side_slope(self):
        return self.linSideSlope

    @lin_side_slope.setter
    def lin_side_slope(self, value):
        self.linSideSlope = value

    @property
    def lin_side_offset(self):
        return self.linSideOffset

    @lin_side_offset.setter
    def lin_side_offset(self, value):
        self.linSideOffset = value

    @property
    def lin_side_break(self):
        return self.linSideBreak

    @lin_side_break.setter
    def lin_side_break(self, value):
        self.linSideBreak = value

    @property
    def linear_slope(self):
        return self.linearSlope

    @linear_slope.setter
    def linear_slope(self, value):
        if value is None:
            self.linearSlope = None
        self.linearSlope = value

    def is_encoding_style(self, style=None):
        style = style or self.style
        return style.lower() in [s.lower() for s in self.lin_to_log_styles]

    def is_decoding_style(self, style=None):
        style = style or self.style
        return style.lower() in [s.lower() for s in self.log_to_lin_styles]

    def _logarithmic_function_factory(
        self,
        lin_side_slope=None,
        lin_side_offset=None,
        log_side_slope=None,
        log_side_offset=None,
        lin_side_break=None,
        linear_slope=None,
        base=None,
        style="log10",
    ):
        # TODO: promote to module level? Make static?
        def _is_decoding_style(s):
            s = style.lower()
            return s.startswith("anti") or s.endswith("lin")

        function_kwargs = {}

        if style[-1] in ["2", "0"]:
            __function = partial(
                logarithmic_function_basic, base=int(style[-1]), style=style
            )

        elif style.startswith("anti") or any(
            [
                x is None
                for x in [
                    lin_side_slope,
                    lin_side_offset,
                    log_side_slope,
                    log_side_offset,
                ]
            ]
        ):
            style = "logB"
            if style.lower().startswith("anti"):
                style = "antiLogB"

            __function = partial(logarithmic_function_basic, base=base, style=style)

        else:
            function_kwargs = dict(
                log_side_slope=log_side_slope,
                log_side_offset=log_side_offset,
                lin_side_slope=lin_side_slope,
                lin_side_offset=lin_side_offset,
            )

            if lin_side_break is not None:
                function_kwargs.update(lin_side_break=lin_side_break)

                if linear_slope is not None:
                    function_kwargs.update(linear_slope=linear_slope)

                style = (
                    "cameraLogToLin" if _is_decoding_style(style) else "cameraLinToLog"
                )
                __function = partial(
                    logarithmic_function_camera, base=base, style=style
                )

            else:
                style = "logToLin" if _is_decoding_style(style) else "linToLog"
                __function = partial(
                    logarithmic_function_quasilog, base=base, style=style
                )

            if any([as_float_array(v).size > 1 for v in function_kwargs.values()]):
                function_kwargs = {
                    k: v * np.ones(3) for k, v in function_kwargs.items()
                }

        return partial(__function, **function_kwargs)

    def _apply_directed(self, RGB, inverse=False):
        RGB_out = as_float_array(RGB)

        inverse_styles = {
            fwd: inv
            for fwd, inv in zip(
                self.lin_to_log_styles + self.log_to_lin_styles,
                self.log_to_lin_styles + self.lin_to_log_styles,
            )
        }
        style = inverse_styles[self.style] if inverse else self.style
        logarithmic_function = self._logarithmic_function_factory(
            style=style,
            base=self.base,
            lin_side_slope=self.lin_side_slope,
            lin_side_offset=self.lin_side_offset,
            log_side_slope=self.log_side_slope,
            log_side_offset=self.log_side_offset,
            lin_side_break=self.lin_side_break,
            linear_slope=self.linear_slope,
        )

        return logarithmic_function(RGB_out)

    def apply(self, RGB, *args):
        return self._apply_directed(RGB, inverse=False)

    def reverse(self, RGB):
        return self._apply_directed(RGB, inverse=True)

    def __str__(self):
        direction = (
            "Log to Linear" if self.style in self.log_to_lin_styles else "Linear to Log"
        )
        title = "{0}{1}".format(
            "{0} - ".format(self.name) if self.name else "", direction
        )
        basic_style = self.style[-1] in "20"
        return (
            "{0} - {1}\n"
            "{2}\n\n"
            "style          : {3}\n"
            "base           : {4}"
            "{5}{6}{7}{8}{9}{10}{11}"
        ).format(
            self.__class__.__name__,
            title,
            "-" * (len(self.__class__.__name__) + 3 + len(title)),
            self.style,
            self.base,
            (
                "\nlogSideSlope   : {0}".format(self.log_side_slope)
                if not basic_style
                else ""
            ),
            (
                "\nlogSideOffset  : {0}".format(self.log_side_offset)
                if not basic_style
                else ""
            ),
            (
                "\nlinSideSlope   : {0}".format(self.lin_side_slope)
                if not basic_style
                else ""
            ),
            (
                "\nlinSideOffset  : {0}".format(self.lin_side_offset)
                if not basic_style
                else ""
            ),
            (
                "\nlinearSlope    : {0}".format(self.linear_slope)
                if not basic_style and self.linear_slope is not None
                else ""
            ),
            (
                "\nlinSideBreak   : {0}".format(self.lin_side_break)
                if not basic_style and self.lin_side_break is not None
                else ""
            ),
            "\n\n{0}".format("\n".join(self.comments)) if self.comments else "",
        )

    def __repr__(self):
        # TODO: show only the used parameters (see __str__ method)
        return (
            "{0}("
            "base={1}, "
            "logSideSlope={2}, "
            "logSideOffset={3}, "
            "linSideSlope={4}, "
            "linSideOffset={5}, "
            "linearSlope={6}, "
            "linSideBreak={7}, "
            'style="{8}"{9})'
        ).format(
            self.__class__.__name__,
            self.base,
            self.log_side_slope,
            self.log_side_offset,
            self.lin_side_slope,
            self.lin_side_offset,
            self.linear_slope,
            self.lin_side_break,
            self.style,
            ', name="{0}"'.format(self.name) if self.name else "",
        )


def shape_cube_lg2(
    function,
    min_exposure=-6.5,
    max_exposure=6.5,
    middle_grey=0.18,
    shaper_size=2**14 - 1,
    cube_size=33,
    name=None,
):
    # 1D shaper
    domain = middle_grey * 2 ** np.array([min_exposure, max_exposure])

    shaper_lut = LUT1D(domain=domain, size=shaper_size)
    shaper_lut.name = "{0} - Shaper".format(name) if name else "Shaper"
    shaper_lut.table = log_encoding_Log2(
        shaper_lut.table,
        middle_grey=middle_grey,
        min_exposure=min_exposure,
        max_exposure=max_exposure,
    )

    # 3D cube
    shaped_cube = LUT3D(size=cube_size, name=name)
    shaped_cube.table = log_decoding_Log2(
        shaped_cube.table,
        middle_grey=middle_grey,
        min_exposure=min_exposure,
        max_exposure=max_exposure,
    )
    shaped_cube.table = function(shaped_cube.table)

    # Concatenate 1D shaper + 3D lut
    lut1d3d = LUTSequence(shaper_lut, shaped_cube)

    return lut1d3d
