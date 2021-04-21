import streamlit
import numpy as np
import colour
from colour.io.luts import AbstractLUTSequenceOperator
import pandas
import helpers


LUMINANCE_WEIGHTS_BT709 = np.array(
    # Minor tweak to green's BT.709 weight to assert sum to unity.
    [
        [0.2126729, 0.2126729, 0.2126729],
        [0.7151521, 0.7151521, 0.7151521],
        [0.0721750, 0.0721750, 0.0721750],
    ]
)


class generic_aesthetic_transfer_function(AbstractLUTSequenceOperator):
    def __init__(
        self,
        contrast=1.0,
        shoulder_contrast=1.0,
        middle_grey_in=0.18,
        middle_grey_out=0.18,
        ev_above_middle_grey=4.0,
    ):
        self._linear = None
        self.set_transfer_details(
            contrast,
            shoulder_contrast,
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
        min_ev = 1.0
        max_ev = 20.0
        ev = np.clip(ev, min_ev, max_ev)
        self._ev_above_middle_grey = ev
        self._radiometric_maximum = np.power(2.0, ev) * self._middle_grey_in

    def set_transfer_details(
        self,
        contrast,
        shoulder_contrast,
        middle_grey_in,
        middle_grey_out,
        ev_above_middle_grey,
    ):
        self._contrast = contrast
        self._shoulder_contrast = shoulder_contrast
        self._middle_grey_in = middle_grey_in
        self._middle_grey_out = middle_grey_out
        self.ev_above_middle_grey = ev_above_middle_grey

        self._shoulder_multiplied = self._contrast * self._shoulder_contrast

        middle_grey_contrast = pow(self._middle_grey_in, self._contrast)

        middle_grey_shoulder_contrast = pow(
            self._middle_grey_in, self._shoulder_multiplied
        )

        radiometric_contrast = pow(self._radiometric_maximum, self._contrast)

        radiometric_multiplied_contrast = pow(
            self._radiometric_maximum, self._shoulder_multiplied
        )

        u = (
            radiometric_multiplied_contrast * self._middle_grey_out
            - middle_grey_shoulder_contrast * self._middle_grey_out
        )
        v = middle_grey_shoulder_contrast * self._middle_grey_out

        self.b = -(
            (
                -middle_grey_contrast
                + (
                    self._middle_grey_out
                    * (
                        radiometric_multiplied_contrast * middle_grey_contrast
                        - radiometric_contrast * v
                    )
                )
                / u
            )
            / v
        )
        self.c = (
            radiometric_multiplied_contrast * middle_grey_contrast
            - radiometric_contrast * v
        ) / u

        self.calculate_LUT()

    def evaluate(self, x):
        x = np.minimum(self._radiometric_maximum, x)
        z = np.ma.power(x, self._contrast).filled(fill_value=0.0)
        y = z / (np.power(z, self._shoulder_contrast) * self.b + self.c)

        return np.asarray(y)

    def calculate_luminance(self, RGB_input):
        return np.ma.dot(RGB_input, LUMINANCE_WEIGHTS_BT709)

    def calculate_maximal_chroma(self, RGB_input):
        return np.ma.divide(
            RGB_input, np.ma.amax(RGB_input, keepdims=True, axis=-1)
        ).filled(fill_value=0.0)

    def compare_RGB(self, RGB_input_A, RGB_input_B):
        return np.ma.all(np.isclose(RGB_input_A, RGB_input_B), axis=-1)

    def evaluate_siragusano2021(self, x):
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
            table=self.evaluate_siragusano2021(
                np.linspace(0.0, self._radiometric_maximum, LUT_size)
            ),
            name="Siragusano Smith 2021",
            domain=[0.0, self._radiometric_maximum + 0.005],
        )
        self._LUT_size = LUT_size
        self._linear = np.linspace(0.0, self._radiometric_maximum, self._LUT_size)

    def apply(self, RGB, per_channel=False, **kwargs):
        if per_channel:
            return self.apply_per_channel(RGB, **kwargs)
        else:
            return self.apply_maxRGB(RGB, **kwargs)

    def apply_maxRGB(self, RGB, gamut_clip=False, gamut_clip_alert=False):
        gamut_clipped_above = np.where(RGB >= self._radiometric_maximum)

        maximum_RGBs = np.amax(RGB, axis=2, keepdims=True)
        ratios = np.ma.divide(RGB, maximum_RGBs).filled(fill_value=0.0)

        curve_evaluation = self.evaluate(maximum_RGBs)

        output_RGBs = curve_evaluation * ratios

        if gamut_clip is True:
            output_RGBs[
                gamut_clipped_above[0], gamut_clipped_above[1], :
            ] = self._LUT.table[-1]

        if gamut_clip_alert is True:
            output_RGBs[gamut_clipped_above[0], gamut_clipped_above[1], :] = [
                1.0,
                0.0,
                0.0,
            ]

        return output_RGBs

    def derive_luminance(self, RGB, gamut_clip=False, gamut_clip_alert=False):
        gamut_clipped_above = np.where(RGB >= self._radiometric_maximum)

        interpolator = colour.LinearInterpolator(
            self._linear,
            self._LUT.table,
        )

        extrapolator = colour.Extrapolator(interpolator)

        maximum_RGBs = np.amax(RGB, axis=-1, keepdims=True)
        ratios = np.ma.divide(RGB, maximum_RGBs).filled(fill_value=0.0)

        luminance_RGBs = self.calculate_luminance(np.abs(ratios))

        # curve_evaluation = self.evaluate(maximum_RGBs)
        curve_evaluation = extrapolator(maximum_RGBs)

        output_RGBs = curve_evaluation * luminance_RGBs

        if gamut_clip is True:
            output_RGBs[
                gamut_clipped_above[0], gamut_clipped_above[1], :
            ] = self._LUT.table[-1]

        if gamut_clip_alert is True:
            output_RGBs[gamut_clipped_above[0], gamut_clipped_above[1], :] = [
                1.0,
                0.0,
                0.0,
            ]

        return output_RGBs

    def luminance_mapping(self, RGB, gamut_clip=False, gamut_clip_alert=False):
        gamut_clipped_above = np.where(RGB >= self._radiometric_maximum)

        luminance_RGBs = self.calculate_luminance(RGB)

        curve_evaluation = self.evaluate(luminance_RGBs)

        output_RGBs = curve_evaluation

        if gamut_clip is True:
            output_RGBs[
                gamut_clipped_above[0], gamut_clipped_above[1], :
            ] = self._LUT.table[-1]

        if gamut_clip_alert is True:
            output_RGBs[gamut_clipped_above[0], gamut_clipped_above[1], :] = [
                1.0,
                0.0,
                0.0,
            ]

        return output_RGBs

    def luminance_difference(self, RGB, gamut_clip=False, gamut_clip_alert=False):
        gamut_clipped_above = np.where(RGB >= self._radiometric_maximum)

        maximum_RGBs = np.amax(RGB, axis=-1, keepdims=True)

        # Out of gamut prism values will be negative after this operation.
        chroma_normalized = np.ma.divide(RGB, maximum_RGBs).filled(fill_value=0.0)

        # print("****** RUN ******")
        # print("Max of chroma_normalized {}".format(np.amax(chroma_normalized)))
        # print("Min of chroma_normalized {}\n".format(np.amin(chroma_normalized)))

        # Abs result to properly sum luminance for values that are outside
        # of the gamut prism.
        luminance_RGBs = self.calculate_luminance(RGB)

        # print("Max of luminance_RGBs {}".format(np.amax(luminance_RGBs)))
        # print("Min of luminance_RGBs {}\n".format(np.amin(luminance_RGBs)))

        # curve_evaluation = self.evaluate(maximum_RGBs)
        luminance_curve_evaluation = self.evaluate(luminance_RGBs)

        # print("Max of the luminance_curve {}".format(np.amax(luminance_curve_evaluation)))
        # print("Min of the luminance_curve {}\n".format(np.amin(luminance_curve_evaluation)))

        # Calculate normalized chroma maximal luminance.
        luminance_chroma_normalized = self.calculate_luminance(chroma_normalized)

        # print("Max of luminance_chroma_normalized {}".format(np.amax(luminance_chroma_normalized)))
        # print("Min of luminance_chroma_normalized {}\n".format(np.amin(luminance_chroma_normalized)))

        # Calculate how much intensity scaling is required to scale the values
        # to the proper luminance for output.
        luminance_scalar = np.ma.divide(
            luminance_curve_evaluation, luminance_chroma_normalized
        ).filled(fill_value=0.0)

        # print("Max of the luminance_scalar {}".format(np.amax(luminance_scalar)))
        # print("Min of the luminance_scalar {}\n".format(np.amin(luminance_scalar)))

        # Scale the RGBs such that they match the output luminance.
        target_RGBs = luminance_scalar * chroma_normalized

        # print("Max of target_RGBs {}".format(np.amax(target_RGBs)))
        # print("Min of target_RGBs {}\n".format(np.amin(target_RGBs)))

        # Here we subtract the theoretical RGB mixtures from the maximal
        # mixtures capable at the display. If the value is negative, the
        # mixture is expressible at the display. If the value is postive,
        # the mixture is beyond the gamut volume. Clip the normalized chroma
        # to avoid gamut prism contribution. Clip and retain only the positive
        # portion.
        RGB_target_diff = np.abs(
            np.clip(
                np.clip(target_RGBs, 0.0, None) - np.clip(chroma_normalized, 0.0, None),
                0.0,
                None,
            )
        )

        # print("Max of RGB_target_diff {}".format(np.amax(RGB_target_diff)))
        # print("Min of RGB_target_diff {}\n".format(np.amin(RGB_target_diff)))

        # Any negative values are outside of the gamut volume.
        # RGB_luminance_target_diff[RGB_luminance_target_diff < 0.0] = 0.0

        # Try to give an idea of how much out of whack the values are by
        # adding achromatic light to the difference view based on how
        # much the difference exceeds the maximum output at the display.
        # diff_diff = (
        #     RGB_luminance_target_diff[..., RGB_luminance_target_diff > 1.0] - 1.0
        # ).reshape((-1, 1))

        # over_difference = np.clip(
        #     RGB_luminance_target_diff[np.any(RGB_luminance_target_diff > 1.0, axis=-1)]
        #     - 1.0,
        #     0.0,
        #     None,
        # )

        # over_difference_luminance = self.derive_luminance(RGB_luminance_target_diff)
        # print("over_difference shape({})".format(over_difference_luminance.shape))

        # print("over sample ({})".format(over_difference_luminance[0, 0, :]))

        # print("np.any shape({})".format(RGB_luminance_target_diff[
        #     np.any(RGB_luminance_target_diff > 1.0, axis=-1)
        # ].shape))
        # RGB_luminance_target_diff[
        #     np.any(RGB_luminance_target_diff > 1.0, axis=-1)
        # ] += over_difference_luminance

        luminance_diff = self.calculate_luminance(RGB_target_diff.copy() - 1.0)

        RGB_target_diff += np.clip(luminance_diff - 1.0, 0.0, None)

        # print("Max of luminance_diff {}".format(np.amax(luminance_diff)))
        # print("Min of luminance_diff {}\n".format(np.amin(luminance_diff)))

        output_RGBs = RGB_target_diff

        if gamut_clip is True:
            output_RGBs[
                gamut_clipped_above[0], gamut_clipped_above[1], :
            ] = self._LUT.table[-1]

        if gamut_clip_alert is True:
            output_RGBs[gamut_clipped_above[0], gamut_clipped_above[1], :] = [
                1.0,
                0.0,
                0.0,
            ]

        return output_RGBs

    def DEVILS_adjust_luminance(self, RGB_input, luminance_output):
        # Calculate the maximal chroma expressible at the display for the incoming
        # RGB triplet.
        maximal_chroma = self.calculate_maximal_chroma(RGB_input)
        # print("maximal_chroma:\n{}".format(maximal_chroma))

        # Calculate the luminance of the maximal chroma expressible at the display.
        maximal_chroma_luminance = self.calculate_luminance(maximal_chroma)
        # print("maximal_chroma_luminance:\n{}".format(maximal_chroma_luminance))

        # Calculate luminance reserves of inverse maximal chroma.
        maximal_reserves = 1.0 - maximal_chroma
        # print("maximal_reserves:\n{}".format(maximal_reserves))

        # Calculate the luminance of the maximal reserves.
        maximal_reserves_luminance = self.calculate_luminance(maximal_reserves)
        # print("maximal_reserves_luminance:\n{}".format(maximal_reserves_luminance))

        # Calculate the difference between the desired output luminance and
        # the maximally chrominant luminance.
        luminance_difference = np.clip(
            luminance_output - maximal_chroma_luminance, 0.0, None
        )
        # print("luminance_difference:\n{}".format(luminance_difference))

        luminance_difference_scalar = np.ma.divide(
            luminance_difference, maximal_reserves_luminance
        ).filled(0.0)
        # print("luminance_difference_scalar:\n{}".format(luminance_difference_scalar))

        chroma_scalar = np.ma.divide(
            luminance_output - luminance_difference, maximal_chroma_luminance
        ).filled(0.0)
        # print("chroma_scalar:\n{}".format(chroma_scalar))

        reserves_compliment = luminance_difference_scalar * maximal_reserves
        # print("reserves_compliment:\n{}".format(reserves_compliment))

        chroma_scaled = chroma_scalar * maximal_chroma
        # print("chroma_scaled:\n{}".format(chroma_scaled))

        # (RGB * scale) + ((1.0 - scale) * luminance)
        return chroma_scaled + reserves_compliment

    def DEVILS_render(self, RGB, gamut_clip=False, gamut_clip_alert=False):
        # Abs result to properly sum luminance for values that are outside
        # of the gamut prism.
        luminance_RGBs = self.calculate_luminance(RGB)

        # curve_evaluation = self.evaluate(maximum_RGBs)
        luminance_curve_evaluation = self.evaluate(luminance_RGBs)

        return self.DEVILS_adjust_luminance(
            np.clip(RGB, 0.0, None), luminance_curve_evaluation
        )

    def channel_clip_render(self, RGB, gamut_clip=False, gamut_clip_alert=False):
        maximum_RGBs = np.amax(RGB, axis=-1, keepdims=True)

        # Out of gamut prism values will be negative after this operation.
        chroma_normalized = np.ma.divide(RGB, maximum_RGBs).filled(fill_value=0.0)

        # Abs result to properly sum luminance for values that are outside
        # of the gamut prism.
        luminance_RGBs = self.calculate_luminance(RGB)

        # curve_evaluation = self.evaluate(maximum_RGBs)
        luminance_curve_evaluation = self.evaluate(luminance_RGBs)

        # Calculate normalized chroma maximal luminance.
        luminance_chroma_normalized = self.calculate_luminance(chroma_normalized)

        # Calculate how much intensity scaling is required to scale the values
        # to the proper luminance for output.
        luminance_scalar = np.ma.divide(
            luminance_curve_evaluation, luminance_chroma_normalized
        ).filled(fill_value=0.0)

        # Scale the RGBs such that they match the output luminance.
        target_RGBs = luminance_scalar * chroma_normalized

        target_gamut_clip_masked = np.ma.masked_greater(target_RGBs, 1.0)

        # Set all values that aren't masked to zero
        target_gamut_clip_masked[~target_gamut_clip_masked.mask] = 0.0

        output_RGBs = target_gamut_clip_masked.filled(fill_value=1.0)

        return output_RGBs

    def apply_per_channel(self, RGBs, gamut_clip=False, gamut_clip_alert=False):
        gamut_clipped_above = np.where(RGBs >= self._radiometric_maximum)
        input_RGBs = np.abs(RGBs)
        evaluate_RGBs = self.evaluate(input_RGBs)

        # Restore negatives that the curve removed the sign from.
        output_RGBs = np.negative(evaluate_RGBs, where=(RGBs < 0.0), out=evaluate_RGBs)

        if gamut_clip is True:
            output_RGBs[
                gamut_clipped_above[0], gamut_clipped_above[1], :
            ] = self._LUT.table[-1]
        if gamut_clip_alert is True:
            output_RGBs[gamut_clipped_above[0], gamut_clipped_above[1], :] = [
                1.0,
                0.0,
                0.0,
            ]

        return output_RGBs


def apply_inverse_EOTF(RGB, EOTF=2.2):
    return np.ma.power(RGB, (1.0 / EOTF)).filled(fill_value=0.0)


def video_buffer(x, exposure_adjustment=0.0):
    return (2.0 ** exposure_adjustment) * x


def application_image_formation_01():
    with streamlit.sidebar:
        exposure_adjustment_help = streamlit.beta_expander("Exposure Adjustment")
        with exposure_adjustment_help:
            streamlit.text_area(
                label="",
                key="Exposure Adjustment Help",
                value="The exposure adjustment in log2 / stops. This "
                "number is a simple multiplier value applied against the "
                "radiometric-like RGB values in the working space. "
                "The slider value becomes the exponent of base two. This value "
                "is used to test for over and under compression of the generated "
                "aestehtic transfer function.",
            )
        exposure_adjustment = streamlit.slider(
            label="",
            key="Exposure Adjustment",
            min_value=-20.0,
            max_value=+20.0,
            value=0.0,
            step=0.25,
        )

        contrast_help = streamlit.beta_expander("Contrast")
        with contrast_help:
            streamlit.text_area(
                label="",
                key="Contrast Help",
                value="The aesthetic contrast of the aesthetic transfer function. "
                "This value controls the overall contrast of the aesthetic "
                "transfer function curve and corresponds to the slope angle "
                "of the linear portion of the curve around middle grey.",
            )
        contrast = streamlit.slider(
            label="",
            key="Contrast",
            min_value=0.01,
            max_value=3.00,
            value=1.20,
            step=0.01,
        )

    region_1_1, region_1_2, region_1_3 = streamlit.beta_columns((2, 5, 2))
    image_region_1_1, image_region_1_2 = streamlit.beta_columns(2)

    with region_1_1:
        middle_grey_input_help = streamlit.beta_expander("Middle Grey Input")
        with middle_grey_input_help:
            streamlit.text_area(
                label="",
                key="Middle Grey Input Help",
                value="The input working space value chosen to represent a middle "
                "perceptual value. This value is mapped to the output value "
                "at the display output below.",
            )
        middle_grey_input = streamlit.number_input(
            label="",  # "Middle Grey Input Value, Radiometric",
            key="Middle Grey Input",
            min_value=0.01,
            max_value=1.0,
            value=0.18,
            step=0.001,
        )

        middle_grey_output_help = streamlit.beta_expander("Middle Grey Output")
        with middle_grey_output_help:
            streamlit.text_area(
                label="",
                key="Middle Grey Output Help",
                value="The output display emission value. This value is the value "
                "the input middle grey value will be mapped to at the display. "
                "The subject of 'middle grey' is one mired in much history and "
                "debate. The default has been chosen based on a perceptual "
                "approximation at 18% emission between maximum and minimum.",
            )
        middle_grey_output = streamlit.number_input(
            label="",
            key="Middle Grey Output",
            min_value=0.01,
            max_value=1.0,
            value=0.18,
            step=0.001,
        )

        maximum_ev_help = streamlit.beta_expander("Maximum EV Above Middle Grey")
        with maximum_ev_help:
            streamlit.text_area(
                label="",
                key="Maximum EV Above Middle Grey Help",
                value="The maximum EV in log2 / stops, above middle grey. This value "
                "is the number of stops above middle grey from the working "
                "space that will be compressed between the middle grey output "
                "and display maximum.",
            )
        maximum_ev = streamlit.slider(
            label="",
            key="Maximum EV Above Middle Grey",
            min_value=1.0,
            max_value=15.0,
            value=10.0,
            step=0.25,
        )

        # Removed Shoulder Contrast, as this has no equivalent parameter
        # under Siragusano Smith 2021. Can re-add it if selectable curve
        # formations are provided.
        #
        # shoulder_contrast_help = streamlit.beta_expander("Shoulder Contrast")
        # with shoulder_contrast_help:
        #     streamlit.text_area(
        #         label="",
        #         key="Shoulder Contrast Help",
        #         value="Shoulder contrast value. This value is an abstract value "
        #         "based on the formula used for the asethetic transfer function. "
        #         "It will control the tension of the curved portion near the "
        #         "maximum display emission.",
        #     )
        # shoulder_contrast = streamlit.slider(
        #     label="",
        #     key="Shoulder Contrast",
        #     min_value=0.61,
        #     max_value=1.00,
        #     value=0.85,
        #     step=0.01,
        # )

    LUT = generic_aesthetic_transfer_function()
    LUT.set_transfer_details(
        contrast=contrast,
        shoulder_contrast=1.0,
        middle_grey_in=middle_grey_input,
        middle_grey_out=middle_grey_output,
        ev_above_middle_grey=maximum_ev,
    )

    with region_1_2:
        plot_data = pandas.DataFrame(
            {"Radiometric": LUT._linear, "Aesthetic Curve": LUT._LUT.table}
        )
        plot_data = plot_data.set_index("Radiometric")
        streamlit.line_chart(data=plot_data, height=200)

        upload_image_help = streamlit.beta_expander("Upload Image")
        with upload_image_help:
            streamlit.text_area(
                label="",
                key="Upload Image Help",
                value="Choose a custom upload image for evaluation. Accepted file "
                "types are .EXR and .HDR.",
            )
        upload_image = streamlit.file_uploader(
            label="", key="Upload Image", type=[".exr", ".hdr"]
        )

        default_image_help = streamlit.beta_expander("Default Test Images")
        with default_image_help:
            streamlit.text_area(
                label="",
                key="Default Test Images Help",
                value="Choose a test image from the defaults for evaluation.",
            )
        default_image_path = streamlit.selectbox(
            label="",
            key="Default Test Images",
            options=[
                "Red Xmas 709",
                "Blue Bar 709",
                "Alexa Models and Charts",
                "Multi-Swatch Test Chart",
                "CC24 Chart, Synthetic",
                "CC24 Chart Photo",
                "CC24 Chart Photo, Cropped",
                "Synthetic Testing Chart",
                "CLF Test Image",
                "Marcie 4K",
            ],
        )

    with region_1_3:
        EOTF_help = streamlit.beta_expander("Display Hardware EOTF")
        with contrast_help:
            streamlit.text_area(
                label="",
                key="EOTF Help",
                value="The display hardware Electro Optical Transfer Function. "
                "This is a technical control that must match the hardware "
                "transfer function of the display used to view this web "
                "page. For most conventional displays such as sRGB-like or "
                "Display P3-like displays, this value should be 2.2. For "
                "broadcast BT.1886 displays, it should be 2.4. Some web "
                "browsers may be using colour management or inconsistent and / "
                "or broken software paths.",
            )
        EOTF = streamlit.number_input(
            label="",
            key="Display Hardware EOTF",
            min_value=1.0,
            max_value=3.0,
            value=2.2,
            step=0.01,
        )

        performance_downscale_help = streamlit.beta_expander("Performance Downscale")
        with performance_downscale_help:
            streamlit.text_area(
                label="",
                key="Performance Downscale Help",
                value="A value to divide the image resolution by, for performance. "
                "Image resolution will be crudely reduced by the multiple "
                "chosen.",
            )
        image_scale = streamlit.slider(
            label="",
            key="Performance Downscale",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
        )

    if upload_image is None:
        default_image_path = helpers.get_dependency_local_path(default_image_path)
        default_image = colour.io.read_image_Imageio(default_image_path)[..., 0:3]
        reduced_image = default_image[::image_scale, ::image_scale, ...]
        original_image = default_image
    else:
        original_image = colour.io.read_image_Imageio(upload_image.read())[..., 0:3]
        reduced_image = original_image[::image_scale, ::image_scale, ...]

    img = reduced_image

    img_max_RGB = LUT.channel_clip_render(
        video_buffer(img, exposure_adjustment),
        True,
        False,
    )
    img_max_RGB_final = apply_inverse_EOTF(img_max_RGB, EOTF)

    img_luminance = LUT.derive_luminance(
        video_buffer(img, exposure_adjustment),
        False,
        False,
    )
    img_luminance_final = apply_inverse_EOTF(img_luminance, EOTF)

    img_map_luminance = LUT.luminance_mapping(
        video_buffer(img, exposure_adjustment),
        False,
        False,
    )
    img_map_luminance_final = apply_inverse_EOTF(img_map_luminance, EOTF)

    img_DEVILS_render = LUT.DEVILS_render(
        video_buffer(img, exposure_adjustment),
        False,
        False,
    )
    img_DEVILS_render_final = apply_inverse_EOTF(img_DEVILS_render, EOTF)

    with image_region_1_1:
        streamlit.image(
            img_max_RGB_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Reconstructed Closed Domain Clipped Values",
        )

        streamlit.image(
            img_map_luminance_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Open Domain Luminance Mapped to Closed Domain",
        )

    with image_region_1_2:
        streamlit.image(
            img_luminance_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Luminance from Traditionl Closed Domain Calculation",
        )

        streamlit.image(
            img_DEVILS_render_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Exposure Value Invariant Luminance Scaling - "
            "Sobotka 2021 / Siragusano Aesthetic Transfer "
            "Function, with Smith Additions 2021 / "
            "Exposure Invariant Gamut Prism Compression Forthcoming",
        )
