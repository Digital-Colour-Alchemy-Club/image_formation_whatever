import numpy as np
import matplotlib
import colour
import streamlit as st
from colour.io.luts import AbstractLUTSequenceOperator

import helpers


class generic_aesthetic_transfer_function(AbstractLUTSequenceOperator):
    def __init__(
        self,
        contrast=1.0,
        shoulder_contrast=1.0,
        middle_grey_in=0.18,
        middle_grey_out=0.18,
        ev_above_middle_grey=4.0,
    ):
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

    def apply(self, RGB, per_channel=False, **kwargs):
        if per_channel:
            return self.apply_maxRGB(RGB, **kwargs)
        else:
            return self.apply_per_channel(RGB, **kwargs)

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

    def apply_per_channel(self, RGB, gamut_clip=False, gamut_clip_alert=False):
        gamut_clipped_above = np.where(RGB >= self._radiometric_maximum)

        output_RGBs = self.evaluate(RGB)

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


# See if this resolves the Cache Crashy.
# @#st.cache(suppress_st_warning=True)
img_path = helpers.get_dependency("Marcie 4K")
default_image = colour.read_image(img_path)[..., 0:3]


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
            label="Contrast", min_value=0.01, max_value=3.00, value=1.75, step=0.01
        )

    region_1_1, region_1_2, region_1_3 = st.beta_columns((2, 5, 2))
    scopes_1, scopes_2, scopes_3, scopes_4 = st.beta_columns((1, 1, 1, 1))
    gamut_col_1_1, gamut_col_1_2, gamut_col_1_3, gamut_col_1_4 = st.beta_columns(
        (1, 1, 1, 1)
    )
    image_region_1_1, image_region_1_2 = st.beta_columns(2)

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
        st.line_chart(data=LUT._LUT.table, height=200)
        upload_image = st.file_uploader(label="Input Image", type=[".exr", ".hdr"])

        default_image_path = st.selectbox(
            label="Test Image",
            options=[
                "Marcie 4K",
                "CC24 Chart Photo",
                "Synthetic Testing Chart",
                "CLF Test Image",
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
        default_image_path = helpers.get_dependency(default_image_path)
        default_image = colour.io.read_image_Imageio(default_image_path)[..., 0:3]
        reduced_image = default_image[::image_scale, ::image_scale, ...]
        original_image = default_image
    else:
        original_image = colour.io.read_image_Imageio(upload_image.read())[..., 0:3]
        reduced_image = original_image[::image_scale, ::image_scale, ...]

    img = reduced_image

    if show_scopes is True:
        scopes_image = img[::plots_scale, ::plots_scale, ...]

        scopes_maxRGB_result = apply_inverse_EOTF(
            LUT.apply_maxRGB(
                video_buffer(scopes_image, exposure_adjustment),
                gamut_clip_1,
                gamut_warn_1,
            ),
            EOTF,
        )
        scopes_per_result = apply_inverse_EOTF(
            LUT.apply_per_channel(
                video_buffer(scopes_image, exposure_adjustment),
                gamut_clip_2,
                gamut_warn_2,
            ),
            EOTF,
        )

        (
            fig_maxRGB_1976,
            ax_maxRGB_1931,
        ) = colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(
            RGB=scopes_maxRGB_result, colourspace="sRGB", show_diagram_colours=False
        )
        (
            fig_maxRGB_1931,
            ax_maxRGB_1931,
        ) = colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
            RGB=scopes_maxRGB_result, colourspace="sRGB", show_diagram_colours=False
        )
        (
            fig_per_1976,
            ax_per_1931,
        ) = colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(
            RGB=scopes_per_result, colourspace="sRGB", show_diagram_colours=False
        )
        (
            fig_per_1931,
            ax_per_1931,
        ) = colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
            RGB=scopes_per_result, colourspace="sRGB", show_diagram_colours=False
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
            apply_inverse_EOTF(
                LUT.apply_maxRGB(
                    video_buffer(img, exposure_adjustment),
                    gamut_clip_1,
                    gamut_warn_1,
                ),
                EOTF,
            ),
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption=LUT._LUT.name,
        )

    with image_region_1_2:
        st.image(
            apply_inverse_EOTF(
                LUT.apply_per_channel(
                    video_buffer(img, exposure_adjustment),
                    gamut_clip_2,
                    gamut_warn_2,
                ),
                EOTF,
            ),
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Generic Per Channel",
        )
