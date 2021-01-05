import numpy as np
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

        curve_evaluation = self.evaluate(RGB)

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


def application_experimental_image_formation():
    LUT = generic_aesthetic_transfer_function()

    # col1, col2 = st.beta_columns([1, 2])
    with st.sidebar:
        upload_image = st.file_uploader(label="Input Image", type=[".exr"])
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
            value=4.0,
            step=0.25,
        )
        exposure = st.slider(
            label="Exposure Adjustment",
            min_value=-10.0,
            max_value=+10.0,
            value=0.0,
            step=0.25,
        )
        contrast = st.slider(
            label="Contrast", min_value=0.01, max_value=3.00, value=1.75, step=0.01
        )
        shoulder_contrast = st.slider(
            label="Shoulder Contrast",
            min_value=0.01,
            max_value=1.00,
            value=1.0,
            step=0.01,
        )
        gamut_clipping = st.checkbox("Gamut Clip to Maximum", value=True)
        gamut_warning = st.checkbox("Exceeds Gamut Indicator")

    def apply_inverse_EOTF(RGB):
        return np.ma.power(RGB, (1.0 / EOTF)).filled(fill_value=0.0)

    # @st.cache
    def video_buffer(x):
        return (2.0 ** exposure) * x

    @st.cache(suppress_st_warning=True)
    def get_marcie():
        img_path = helpers.get_dependency("Marcie 4K")
        img = colour.read_image(img_path)[..., 0:3]
        return img

    # with col2:
    LUT.set_transfer_details(
        contrast=contrast,
        shoulder_contrast=shoulder_contrast,
        middle_grey_in=middle_grey_input,
        middle_grey_out=middle_grey_output,
        ev_above_middle_grey=maximum_ev,
    )
    st.line_chart(data=LUT._LUT.table)

    if upload_image is not None:
        img = colour.read_image(upload_image.read())
    else:
        img = get_marcie()

    st.image(
        apply_inverse_EOTF(
            LUT.apply_maxRGB(video_buffer(img), gamut_clipping, gamut_warning)
        ),
        clamp=[0.0, 1.0],
        use_column_width=True,
        caption=LUT._LUT.name,
    )

    st.image(
        apply_inverse_EOTF(
            LUT.apply_per_channel(video_buffer(img), gamut_clipping, gamut_warning)
        ),
        clamp=[0.0, 1.0],
        use_column_width=True,
        caption="Generic Per Channel",
    )
