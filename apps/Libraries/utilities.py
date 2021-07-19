import numpy as np
import colour
import colour.io.luts

CCTF_ENCODINGS = np.array(
    [
        "Gamma 2.2",
        "Gamma 2.4",
        "Gamma 2.6",
        "sRGB",
        # "ProPhoto RGB", "RIMM RGB", "ROMM RGB"
        # "ACEScc", "ACEScct", "ACESproxy", "ALEXA Log C", "Canon Log 2", "Canon Log 3"
        # "Canon Log", "Cineon", "D-Log", "ERIMM RGB", "F-Log", "Filmic Pro 6", "Log3G10"
        # "Log3G12", "Panalog", "PLog", "Protune", "REDLog", "REDLogFilm", "S-Log", "S-Log2"
        # "S-Log3", "T-Log", "V-Log", "ViperLog", "ARIB STD-B67", "ITU-R BT.2020"
        # "ITU-R BT.2100 HLG", "ITU-R BT.2100 PQ", "ITU-R BT.601", "ITU-R BT.709"
        # "SMPTE 240M", "DCDM", "DICOM GSDF", "ITU-R BT.1886", "ST 2084"
    ]
)


def enumerate_cctf_encodings():
    return CCTF_ENCODINGS


def apply_cctf_encoding(RGB_input, transfer_function_name="Gamma 2.2", exponent=None):
    return colour.cctf_encoding(
        RGB_input, function=transfer_function_name, negative_number_handling="Clamp"
    )


def calculate_eiY(XYZ_D65):
    X, Y, Z = np.split(XYZ_D65, 3, axis=-1)

    # CIE 1976 u' v' is a suitable weighting candidate for the
    # constraints detailed above. It is also trivial from
    # an implementation standpoint.
    #
    # This model is also utilized in Nayatani's work
    # Simple estimation methods for the Helmholtz—Kohlrausch effect
    # https://doi.org/10.1002/(SICI)1520-6378(199712)22:6<385::AID-COL6>3.0.CO;2-R
    ei_dividend = np.abs(X) + (15.0 * np.abs(Y)) + (3.0 * np.abs(Z))
    e = np.ma.divide((4.0 * X), ei_dividend).filled(fill_value=0.0)
    i = np.ma.divide((9.0 * Y), ei_dividend).filled(fill_value=0.0)

    eiY_stack = np.vstack([e.T, i.T, Y.T]).T.reshape(XYZ_D65.shape)

    return eiY_stack


def calculate_EVILS_eiY(XYZ_D65, xy_achromatic=[0.3127, 0.3290]):
    # EVILS eiY Oppenent Chroma Model. Fancy words for transforming the
    # CIE XYZ values into u' v' Cartesian coordinates. The true "magic" if
    # there is any, is in the identification of the HK effect issues with
    # the Cartesian distances offered from u' v'.
    e, i, Y = np.split(calculate_eiY(XYZ_D65), 3, axis=-1)
    e_a, i_a, Y_a = np.split(calculate_eiY(colour.xy_to_XYZ(xy_achromatic)), 3, axis=-1)

    eiY_output_e = e - e_a
    eiY_output_i = i - i_a
    eiY_output_Y = Y

    eiY_output = np.vstack([eiY_output_e.T, eiY_output_i.T, eiY_output_Y.T]).T.reshape(
        XYZ_D65.shape
    )

    return eiY_output


def calculate_EVILS_eiHCY(EVILS_eiY):
    # Convert the eiY Cartesian model space into a polar coordinate, with
    # Y as height. Again, none of this is really used for the gamut
    # compression, but simply for shaping. The polar coordinate measurements
    # are strictly used to perform scaling of chroma. That is, no inversions
    # are required.
    e, i, eiY = np.split(EVILS_eiY, 3, axis=-1)
    eiH = np.arctan2(i, e)
    eiC = (e ** 2.0 + i ** 2.0) ** (1.0 / 2.0)

    return np.vstack([eiH.T, eiC.T, eiY.T]).T.reshape(EVILS_eiY.shape)


def power_compression(
    X_input, X_maximum, Y_maximum, compression_power=None, identity_breakpoint=None
):
    # X_maximum - the maximal value input of X where X crosses Y.
    # Y_maximum - the maximal value output of Y.
    # compression_power - degree of compression from zero to infinity, where
    #     higher values push the curve closer to the Y_maximum limit.
    # identity_breakpoint - value below which values will remain unchanged.

    # The workhorse function. Thanks to Jed Smith (again) for applying his
    # rigourous attention to deriving the elegant intersectins and such.

    if identity_breakpoint is None:
        # By default, for sanity of compression, let the entire chroma range
        # elegantly be scaled down ever so slightly. This helps to prevent
        # chroma "kinks" up the range, as everything would be compressed
        # toward this value.
        identity_breakpoint = 0.0

    if compression_power is None:
        # Set to a sane default for chrominance compression using BT.709
        # assumptions.
        compression_power = 1.5

    scale = (X_maximum - identity_breakpoint) / (
        ((Y_maximum - identity_breakpoint) / (X_maximum - identity_breakpoint))
        ** (-compression_power)
        - 1.0
    ) ** (1.0 / compression_power)

    X_minus_breakpoint = X_input - identity_breakpoint

    compressed_out = identity_breakpoint + (
        scale
        * (X_minus_breakpoint / scale)
        / (1.0 + (X_minus_breakpoint / scale) ** compression_power)
        ** (1.0 / compression_power)
    )

    return compressed_out


def calculate_luminance(RGB_input, colourspace_name="HK BT.709 weighting"):
    if colourspace_name == "HK BT.709 weighting":
        XYZ_matrix = np.array(
            [
                [0.13913043, 0.73043478, 0.13043478],
                [0.13913043, 0.73043478, 0.13043478],
                [0.13913043, 0.73043478, 0.13043478],
            ]
        )
    else:
        XYZ_matrix = colour.RGB_COLOURSPACES[colourspace_name].matrix_RGB_to_XYZ

    # luminance_matrix = np.tile(XYZ_matrix[1], (3, 1)).T.reshape((3, 3))
    XYZ = np.clip(np.ma.dot(RGB_input, XYZ_matrix).filled(fill_value=0.0), 0.0, None)
    luminance = np.dstack([XYZ[..., 1]] * 3)

    return luminance


def calculate_maximal_chroma(RGB_input):
    return np.ma.divide(
        RGB_input, np.ma.amax(RGB_input, keepdims=True, axis=-1)
    ).filled(fill_value=0.0)


def compare_RGB(RGB_input_A, RGB_input_B):
    return np.ma.all(np.isclose(RGB_input_A, RGB_input_B), axis=-1)


def calculate_ratio(start, stop, ratio):
    return start + ((stop - start) * ratio)


def adjust_chroma(RGB_input, ratio):
    # The very familiar and poorly named function of yore, here properly
    # named. Chroma is distance from axis. Why is this function so simple
    # and yet luminance invariant? Because in terms of emission energy,
    # the RGB luminance is processed in meatspace inside our perceptual
    # systems. By scaling those lights uniformly, we scale the luminance
    # uniformly. To suppliment the scaled loss in luminance, we simply add
    # back achromatic R=G=B light, and presto... luminance invariant output.
    return (RGB_input * ratio) + ((1.0 - ratio) * calculate_luminance(RGB_input))


def adjust_exposure(RGB_input, exposure_adjustment):
    return (2.0 ** exposure_adjustment) * RGB_input


def calculate_EVILS_LICH(RGB_input, luminance_output):
    # Note that luminance here is the closed domain range, and relative to
    # the closed domain of the working domain. For example, 1.0 would be maximal
    # luminance in both SDR and EDR cases, and care must be taken to assert that
    # the meaning of the luminance chosen matches assumptions.

    # Calculate the maximal chroma expressible at the display for the incoming
    # RGB triplet.
    maximal_chroma = calculate_maximal_chroma(RGB_input)

    # Calculate the luminance of the maximal chroma expressible at the display.
    # TODO: Honour the image encoding's primaries.
    maximal_chroma_luminance = calculate_luminance(
        maximal_chroma, colourspace_name="sRGB"
    )

    # Calculate luminance reserves of inverse maximal chroma.
    maximal_reserves = 1.0 - maximal_chroma

    # Calculate the luminance of the maximal reserves.
    # TODO: Honour the image encoding's primaries.
    maximal_reserves_luminance = calculate_luminance(
        maximal_reserves, colourspace_name="sRGB"
    )

    # Calculate the difference between the desired output luminance and
    # the maximally chrominant luminance.
    luminance_difference = np.clip(
        luminance_output - maximal_chroma_luminance, 0.0, None
    )

    luminance_difference_scalar = np.ma.divide(
        luminance_difference, maximal_reserves_luminance
    ).filled(0.0)

    chroma_scalar = np.ma.divide(
        luminance_output - luminance_difference, maximal_chroma_luminance
    ).filled(0.0)

    reserves_compliment = luminance_difference_scalar * maximal_reserves

    chroma_scaled = chroma_scalar * maximal_chroma

    return chroma_scaled + reserves_compliment


def calculate_EVILS_CLAW(
    RGB_input,
    CLAW_compression=1.0,
    CLAW_identity_limit=None,
    CLAW_maximum_input=None,
    CLAW_maximum_output=None,
):
    XYZ_RGB = colour.sRGB_to_XYZ(RGB_input)
    EVILS_eiY = calculate_EVILS_eiY(XYZ_RGB)
    eiH, eiC, eiY = np.split(calculate_EVILS_eiHCY(EVILS_eiY), 3, axis=-1)

    eiC_max = np.amax(eiC)
    # eiC_n = np.ma.divide(
    #     eiC,
    #     eiC_max
    # ).filled(fill_value=0.0)

    if CLAW_maximum_input is None:
        # Set CLAW_maximum_input to a reasonable portion of the output
        # range. Scalar.
        CLAW_maximum_input = eiC_max
    else:
        CLAW_maximum_input *= eiC_max

    if CLAW_maximum_output is None:
        # Set CLAW_maximum_output to the input to assure that a majority of
        # values are unchanged. Scalar
        CLAW_maximum_output = CLAW_maximum_input * 0.6
    else:
        CLAW_maximum_output *= CLAW_maximum_input

    if CLAW_identity_limit is None:
        # Set breakpoint to a reasonably high value to leave a majority
        # of values untouched.
        CLAW_identity_limit = 0.0
    else:
        CLAW_identity_limit *= CLAW_maximum_output

    EVILS_CLAW_ratio = np.where(
        eiC > CLAW_identity_limit,
        power_compression(
            X_input=eiC,
            X_maximum=CLAW_maximum_input,
            Y_maximum=CLAW_maximum_output,
            compression_power=CLAW_compression,
            identity_breakpoint=CLAW_identity_limit,
        ),
        eiC,
    )

    # Chroma scale here is relative to what the adjust_chroma tool expects.
    # So here we assume that maximal chroma is 100%, and simply subtract
    # the normalized output of the input chroma from the compressed signal
    # chroma. This scaling could likely be improved.
    chroma_scalar = EVILS_CLAW_ratio / eiC

    return adjust_chroma(RGB_input, chroma_scalar)


class generic_aesthetic_transfer_function(colour.io.luts.AbstractLUTSequenceOperator):
    def __init__(
        self,
        contrast=1.0,
        middle_grey_in=0.18,
        middle_grey_out=0.18,
        ev_above_middle_grey=4.0,
    ):
        self._linear = None
        self.set_transfer_details(
            contrast,
            middle_grey_in,
            middle_grey_out,
            ev_above_middle_grey,
        )
        self.calculate_LUT()

    @property
    def ev_above_middle_grey(self):
        return self._ev_above_middle_grey

    @ev_above_middle_grey.setter
    def ev_above_middle_grey(self, ev):
        self._ev_above_middle_grey = ev
        self._radiometric_maximum = np.power(2.0, ev) * self._middle_grey_in

    def set_transfer_details(
        self,
        contrast,
        middle_grey_in,
        middle_grey_out,
        ev_above_middle_grey,
    ):
        self._contrast = contrast
        self._middle_grey_in = middle_grey_in
        self._middle_grey_out = middle_grey_out
        self.ev_above_middle_grey = ev_above_middle_grey

        self.calculate_LUT()

    def evaluate(self, x):
        input_domain_scale = (
            (self._radiometric_maximum * self._middle_grey_in)
            * (self._middle_grey_out ** (1.0 / self._contrast) - 1.0)
        ) / (
            self._middle_grey_in
            - (
                self._radiometric_maximum
                * self._middle_grey_out ** (1.0 / self._contrast)
            )
        )

        output_domain_scale = (
            self._radiometric_maximum / (self._radiometric_maximum + input_domain_scale)
        ) ** -self._contrast

        # Siragusano Smith 2021
        return output_domain_scale * (x / (x + input_domain_scale)) ** self._contrast

    def calculate_LUT(self, LUT_size=1024):
        self._LUT = colour.LUT1D(
            table=self.evaluate(np.linspace(0.0, self._radiometric_maximum, LUT_size)),
            name="Siragusano Smith 2021",
            domain=[0.0, self._radiometric_maximum + 0.005],
        )
        self._LUT_size = LUT_size
        self._linear = np.linspace(0.0, self._radiometric_maximum, self._LUT_size)

    def calculate_luminance_mapping(self, RGB):
        curve_evaluation = self.evaluate(calculate_luminance(RGB))

        output_RGBs = curve_evaluation

        return output_RGBs

    def apply(self, RGB, **kwargs):
        return calculate_EVILS_LICH(RGB, **kwargs)
