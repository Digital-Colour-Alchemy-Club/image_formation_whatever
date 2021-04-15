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
        return np.dot(np.abs(RGB_input), LUMINANCE_WEIGHTS_BT709)

    def calculate_LUT(self, LUT_size=1024):
        self._LUT = colour.LUT1D(
            table=self.evaluate(np.linspace(0.0, self._radiometric_maximum, LUT_size)),
            name="Generic Lottes 2016 with Fixes",
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
            value=1.40,
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
            value=4.50,
            step=0.25,
        )

        shoulder_contrast_help = streamlit.beta_expander("Shoulder Contrast")
        with shoulder_contrast_help:
            streamlit.text_area(
                label="",
                key="Shoulder Contrast Help",
                value="Shoulder contrast value. This value is an abstract value "
                "based on the formula used for the asethetic transfer function. "
                "It will control the tension of the curved portion near the "
                "maximum display emission.",
            )
        shoulder_contrast = streamlit.slider(
            label="",
            key="Shoulder Contrast",
            min_value=0.61,
            max_value=1.00,
            value=0.85,
            step=0.01,
        )

    LUT = generic_aesthetic_transfer_function()
    LUT.set_transfer_details(
        contrast=contrast,
        shoulder_contrast=shoulder_contrast,
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
                "Alexa Models and Charts",
                "Red Xmas 709",
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

    img_max_RGB = LUT.apply_maxRGB(
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

    with image_region_1_1:
        streamlit.image(
            img_max_RGB_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Chroma",
        )

    with image_region_1_2:
        streamlit.image(
            img_luminance_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Luminance from Radiometric-like Open Domain",
        )

        streamlit.image(
            img_map_luminance_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Luminance Mapped from Luminance",
        )
