import streamlit
import altair
import numpy as np
import colour
from colour.io.luts import AbstractLUTSequenceOperator
import pandas
from image_formation_apps import helpers

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
        middle_grey_in=0.18,
        middle_grey_out=0.18,
        ev_above_middle_grey=4.0,
        luminance_weights=LUMINANCE_WEIGHTS_BT709,
    ):
        self._linear = None
        self.set_transfer_details(
            contrast,
            middle_grey_in,
            middle_grey_out,
            ev_above_middle_grey,
        )
        self.calculate_LUT()
        self._luminance_weights = luminance_weights

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
        middle_grey_in,
        middle_grey_out,
        ev_above_middle_grey,
    ):
        self._contrast = contrast
        self._middle_grey_in = middle_grey_in
        self._middle_grey_out = middle_grey_out
        self.ev_above_middle_grey = ev_above_middle_grey

        self.calculate_LUT()

    def calculate_luminance(
        self,
        RGB_input,
    ):
        return np.ma.dot(RGB_input, self._luminance_weights)

    def calculate_maximal_chroma(self, RGB_input):
        return np.ma.divide(
            RGB_input, np.ma.amax(RGB_input, keepdims=True, axis=-1)
        ).filled(fill_value=0.0)

    def compare_RGB(self, RGB_input_A, RGB_input_B):
        return np.ma.all(np.isclose(RGB_input_A, RGB_input_B), axis=-1)

    def calculate_ratio(self, start, stop, ratio):
        return start + ((stop - start) * ratio)

    def adjust_weights(self, bias, entry_weights=LUMINANCE_WEIGHTS_BT709):
        self._luminance_weights = self.calculate_ratio(
            entry_weights, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], bias
        )
        return self._luminance_weights

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

    def apply(self, RGB, **kwargs):
        return self.EVILS_render(RGB, **kwargs)

    def luminance_mapping(self, RGB, gamut_clip=False, gamut_clip_alert=False):
        gamut_clipped_above = np.where(RGB >= self._radiometric_maximum)

        luminance_RGBs = self.calculate_luminance(RGB)

        curve_evaluation = self.evaluate(luminance_RGBs)

        output_RGBs = curve_evaluation

        return output_RGBs

    def EVILS_calculate_RGB_from_luminance(self, RGB_input, luminance_output):
        # Calculate the maximal chroma expressible at the display for the incoming
        # RGB triplet.
        maximal_chroma = self.calculate_maximal_chroma(RGB_input)

        # Calculate the luminance of the maximal chroma expressible at the display.
        maximal_chroma_luminance = self.calculate_luminance(maximal_chroma)

        # Calculate luminance reserves of inverse maximal chroma.
        maximal_reserves = 1.0 - maximal_chroma

        # Calculate the luminance of the maximal reserves.
        maximal_reserves_luminance = self.calculate_luminance(maximal_reserves)

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

    def EVILS_render(self, RGB, gamut_clip=False, gamut_clip_alert=False):
        # Abs result to properly sum luminance for values that are outside
        # of the gamut prism.
        luminance_RGBs = self.calculate_luminance(RGB)

        # curve_evaluation = self.evaluate(maximum_RGBs)
        luminance_curve_evaluation = self.evaluate(luminance_RGBs)

        return self.EVILS_calculate_RGB_from_luminance(
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


def apply_inverse_EOTF(RGB, EOTF=2.2):
    return np.ma.power(RGB, (1.0 / EOTF)).filled(fill_value=0.0)


def video_buffer(x, exposure_adjustment=0.0):
    return (2.0**exposure_adjustment) * x


def application_image_formation_01():
    with streamlit.sidebar:
        exposure_adjustment_help = streamlit.expander("Exposure Adjustment")
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

        contrast_help = streamlit.expander("Contrast")
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

    region_1_1, region_1_2, region_1_3 = streamlit.columns((2, 5, 2))
    image_region_1_1, image_region_1_2 = streamlit.columns(2)

    with region_1_1:
        middle_grey_input_help = streamlit.expander("Middle Grey Input")
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

        middle_grey_output_help = streamlit.expander("Middle Grey Output")
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

        maximum_ev_help = streamlit.expander("Maximum EV Above Middle Grey")
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
            max_value=25.0,
            value=10.0,
            step=0.25,
        )

        # bias_weights_help = streamlit.expander("Bias Luminance Weights")
        # with bias_weights_help:
        #     streamlit.text_area(
        #         label="",
        #         key="Bias Luminance Weights Help",
        #         value="Percentage to bias the luminance weights toward their "
        #         "mean. A small bias can help to address some psychophysical "
        #         "effects such as the Helmholtz-Kohlrausch effect.",
        #     )
        #     bias_weights = streamlit.slider(
        #         label="",
        #         key="Bias Weights",
        #         min_value=0.0,
        #         max_value=1.0,
        #         value=0.0,
        #         step=0.01,
        #     )

    LUT = generic_aesthetic_transfer_function()
    LUT.set_transfer_details(
        contrast=contrast,
        middle_grey_in=middle_grey_input,
        middle_grey_out=middle_grey_output,
        ev_above_middle_grey=maximum_ev,
    )

    with region_1_2:
        x_axis = "Open Domain (Log2 Scale)"
        y_axis = "Closed Domain (Log2 Scale)"
        plot_data = pandas.DataFrame({x_axis: LUT._linear, y_axis: LUT._LUT.table})
        # plot_data = plot_data.set_index("Open Domain")
        chart = (
            altair.Chart(plot_data)
            .transform_filter((altair.datum[x_axis] > 0) & (altair.datum[y_axis] > 0))
            .mark_line()
            .encode(
                x=altair.X(x_axis + ":Q", scale=altair.Scale(type="log", base=2)),
                y=altair.Y(y_axis + ":Q", scale=altair.Scale(type="log", base=2)),
            )
        )

        streamlit.altair_chart(chart, use_container_width=True)
        # streamlit.line_chart(data=plot_data, height=200)

        upload_image_help = streamlit.expander("Upload Image")
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

        default_image_help = streamlit.expander("Default Test Images")
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
        EOTF_help = streamlit.expander("Display Hardware EOTF")
        with EOTF_help:
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

        performance_downscale_help = streamlit.expander("Performance Downscale")
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

    # global_weights = LUT.adjust_weights(bias_weights)

    if upload_image is None:
        default_image_path = helpers.get_dependency_local_path(default_image_path)
        default_image = colour.io.read_image_Imageio(default_image_path)[..., 0:3]
        reduced_image = default_image[::image_scale, ::image_scale, ...]
        original_image = default_image
    else:
        original_image = colour.io.read_image_Imageio(upload_image.read())[..., 0:3]
        reduced_image = original_image[::image_scale, ::image_scale, ...]

    img = reduced_image

    img_clipped_chroma = LUT.channel_clip_render(
        video_buffer(img, exposure_adjustment),
        True,
        False,
    )
    img_clipped_chroma_final = apply_inverse_EOTF(img_clipped_chroma, EOTF)

    img_map_luminance = LUT.luminance_mapping(
        video_buffer(img, exposure_adjustment),
        False,
        False,
    )
    img_map_luminance_final = apply_inverse_EOTF(img_map_luminance, EOTF)

    img_DEVILS_render = LUT.EVILS_render(
        video_buffer(img, exposure_adjustment),
        False,
        False,
    )
    img_EVILS_render_final = apply_inverse_EOTF(img_DEVILS_render, EOTF)

    with image_region_1_1:
        streamlit.image(
            img_map_luminance_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Open Domain Luminance Mapped to Closed Domain",
        )
        streamlit.image(
            img_clipped_chroma_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Reconstructed Closed Domain RGB Clipped Values",
        )

    with image_region_1_2:
        streamlit.image(
            img_EVILS_render_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Exposure Value Invariant Luminance Scaling - "
            "Sobotka 2021 / Siragusano Aesthetic Transfer "
            "Function, with Smith Additions 2021 / "
            "Exposure Invariant Gamut Prism Compression Forthcoming",
        )
