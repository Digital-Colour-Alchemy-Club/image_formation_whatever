import numpy as np
import matplotlib
import colour
import streamlit as st
from colour.io.luts import AbstractLUTSequenceOperator
import pandas
from image_formation_apps import helpers


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
            output_RGBs[gamut_clipped_above[0], gamut_clipped_above[1], :] = (
                self._LUT.table[-1]
            )

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
            output_RGBs[gamut_clipped_above[0], gamut_clipped_above[1], :] = (
                self._LUT.table[-1]
            )
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
    return (2.0**exposure_adjustment) * x


def apply_CDL(in_RGB, slope, offset, power, pivot):
    slope = np.asarray(slope)
    offset = np.asarray(offset)
    power = np.asarray(power)

    return (((slope * (in_RGB / pivot)) + offset) ** power) * pivot


matplotlib.pyplot.style.use({"figure.figsize": (4, 4), "font.size": 4})


def application_experimental_image_formation_00():
    LUT = generic_aesthetic_transfer_function()

    with st.sidebar:
        exposure_adjustment = st.slider(
            label="Exposure Adjustment",
            min_value=-10.0,
            max_value=+10.0,
            value=0.0,
            step=0.25,
        )
        contrast = st.slider(
            label="Contrast", min_value=0.01, max_value=3.00, value=1.25, step=0.01
        )

    region_1_1, region_1_2, region_1_3 = st.columns((2, 5, 2))
    scopes_1, scopes_2, scopes_3, scopes_4 = st.columns((1, 1, 1, 1))
    gamut_col_1_1, gamut_col_1_2, gamut_col_1_3, gamut_col_1_4 = st.columns(
        (1, 1, 1, 1)
    )
    image_region_1_1, image_region_1_2 = st.columns(2)
    with st.expander(label="In Progress Incomplete CDL..."):
        chain_CDLs = st.checkbox("Use Single CDL Control", value=True)
        CDL_A, CDL_B = st.columns(2)

    with region_1_1:
        EOTF = st.number_input(
            label="Display Hardware EOTF",
            min_value=1.0,
            max_value=3.0,
            value=2.2,
            step=0.01,
        )
        middle_grey_input = st.number_input(
            label="Middle Grey Input Value, Radiometric",
            min_value=0.01,
            max_value=1.0,
            value=0.18,
            step=0.001,
        )
        middle_grey_output = st.number_input(
            label="Middle Grey Output Display Value, Radiometric",
            min_value=0.01,
            max_value=1.0,
            value=0.18,
            step=0.001,
        )
        maximum_ev = st.slider(
            label="Maximum EV Above Middle Grey",
            min_value=1.0,
            max_value=15.0,
            value=3.25,
            step=0.25,
        )
        shoulder_contrast = st.slider(
            label="Shoulder Contrast",
            min_value=0.01,
            max_value=1.00,
            value=1.0,
            step=0.01,
        )

    LUT.set_transfer_details(
        contrast=contrast,
        shoulder_contrast=shoulder_contrast,
        middle_grey_in=middle_grey_input,
        middle_grey_out=middle_grey_output,
        ev_above_middle_grey=maximum_ev,
    )
    with region_1_2:
        plot_data = pandas.DataFrame(
            {"Radiometric": LUT._linear, "Curve": LUT._LUT.table}
        )
        plot_data = plot_data.set_index("Radiometric")
        st.line_chart(data=plot_data, height=200)
        upload_image = st.file_uploader(label="Input Image", type=[".exr", ".hdr"])

        default_image_path = st.selectbox(
            label="Test Image",
            options=[
                "Blue Bar",
                "Blue Bar 709",
                "Red Xmas",
                "Red Xmas 709",
                "Okja",
                "Club DJ",
                "Products",
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
        image_scale = st.slider(
            label="Divide Image Resolution By",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
        )
        plots_scale = st.slider(
            label="Divide Plots Resolution By",
            min_value=2,
            max_value=30,
            value=10,
            step=1,
        )
        show_scopes = st.checkbox("Show Scopes and Plots", value=False)

    with gamut_col_1_1:
        gamut_clip_1 = st.checkbox(
            "Formation A: Gamut Clip to Curve Radiometric Maximum", value=True
        )
    with gamut_col_1_2:
        gamut_warn_1 = st.checkbox("Formation A: Exceeds Radiometric Gamut Indicator")

    with gamut_col_1_3:
        gamut_clip_2 = st.checkbox(
            "Formation B: Gamut Clip to Curve Radiometric Maximum", value=False
        )
    with gamut_col_1_4:
        gamut_warn_2 = st.checkbox("Formation B: Exceeds Radiometric Gamut Indicator")

    if upload_image is None:
        default_image_path = helpers.get_dependency_local_path(default_image_path)
        default_image = colour.io.read_image_Imageio(default_image_path)[..., 0:3]
        reduced_image = default_image[::image_scale, ::image_scale, ...]
        original_image = default_image
    else:
        original_image = colour.io.read_image_Imageio(upload_image.read())[..., 0:3]
        reduced_image = original_image[::image_scale, ::image_scale, ...]

    default_slope_A = default_power_A = default_slope_B = default_power_B = 1.0
    default_pivot_A = default_pivot_B = 0.18
    default_offset_A = default_offset_B = 0.0

    with CDL_A:
        slope_B = slope_A = st.slider(
            label="A: Slope",
            min_value=0.01,
            max_value=10.0,
            value=default_slope_A,
            step=0.01,
        )

        offset_B = offset_A = st.number_input(
            label="A: Offset",
            min_value=-5.0,
            max_value=5.0,
            value=default_offset_A,
            step=0.0001,
            format="%.4f",
        )

        power_B = power_A = st.slider(
            label="A: Power",
            min_value=0.01,
            max_value=5.0,
            value=default_power_A,
            step=0.01,
        )

        pivot_B = pivot_A = st.slider(
            label="A: Pivot",
            min_value=0.01,
            max_value=5.0,
            value=default_pivot_A,
            step=0.01,
        )

    if chain_CDLs is False:
        with CDL_B:
            slope_B = st.slider(
                label="B: Slope",
                min_value=0.01,
                max_value=10.0,
                value=default_slope_B,
                step=0.01,
            )

            offset_B = st.number_input(
                label="B: Offset",
                min_value=-5.0,
                max_value=5.0,
                value=default_offset_B,
                step=0.0001,
                format="%.4f",
            )

            power_B = st.slider(
                label="B: Power",
                min_value=0.01,
                max_value=5.0,
                value=default_power_B,
                step=0.01,
            )

            pivot_B = st.slider(
                label="B: Pivot",
                min_value=0.01,
                max_value=5.0,
                value=default_pivot_B,
                step=0.01,
            )

    img = reduced_image
    img_max_RGB = apply_CDL(
        LUT.apply_maxRGB(
            video_buffer(img, exposure_adjustment),
            gamut_clip_1,
            gamut_warn_1,
        ),
        slope_A,
        offset_A,
        power_A,
        pivot_A,
    )
    img_max_RGB_final = apply_inverse_EOTF(img_max_RGB, EOTF)

    img_per_channel = apply_CDL(
        LUT.apply_per_channel(
            video_buffer(img, exposure_adjustment),
            gamut_clip_2,
            gamut_warn_2,
        ),
        slope_B,
        offset_B,
        power_B,
        pivot_B,
    )
    img_per_channel_final = apply_inverse_EOTF(img_per_channel, EOTF)

    if show_scopes is True:
        scopes_image_max_RGB = img_max_RGB[::plots_scale, ::plots_scale, ...]
        scopes_image_per_channel = img_per_channel[::plots_scale, ::plots_scale, ...]

        (
            fig_maxRGB_1976,
            ax_maxRGB_1931,
        ) = colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(
            RGB=scopes_image_max_RGB, colourspace="sRGB", show_diagram_colours=False
        )
        (
            fig_maxRGB_1931,
            ax_maxRGB_1931,
        ) = colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
            RGB=scopes_image_max_RGB, colourspace="sRGB", show_diagram_colours=False
        )
        (
            fig_per_1976,
            ax_per_1976,
        ) = colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(
            RGB=scopes_image_per_channel, colourspace="sRGB", show_diagram_colours=False
        )
        (
            fig_per_1931,
            ax_per_1931,
        ) = colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
            RGB=scopes_image_per_channel, colourspace="sRGB", show_diagram_colours=False
        )

        with scopes_1:
            st.pyplot(fig=fig_maxRGB_1976)
        with scopes_2:
            st.pyplot(fig=fig_maxRGB_1931)

        with scopes_3:
            st.pyplot(fig=fig_per_1976)
        with scopes_4:
            st.pyplot(fig=fig_per_1931)

    with image_region_1_1:
        st.image(
            img_max_RGB_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Maximum RGB Channel Lookup",
        )

    with image_region_1_2:
        st.image(
            img_per_channel_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Per RGB Channel Lookup",
        )
