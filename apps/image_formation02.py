import apps.Libraries.utilities as utilities
import streamlit
import altair
import numpy as np
import colour
import pandas
import helpers


def application_image_formation_02():
    streamlit.markdown(
        "# Exposure Value Invariant Luminance System (EVILS)\n"
        "## An Image Maker First Pixel Processing Pipe\n"
        "---"
    )
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
            value=1.50,
            step=0.01,
        )

        EVILS_CLAW_help = streamlit.beta_expander("EVILS CLAW")
        with EVILS_CLAW_help:
            streamlit.text_area(
                label="",
                key="EVILS CLAW Help",
                value="EVILS Chromaticity Linear Adjustment Hammer. "
                "CLAW is the chroma adjustment tool for use with the EVILS "
                "system. It can be used to perform gamut compression, as well "
                "as intra-gamut compression to correct for the Helmholtz–"
                "Kohlrausch effect.\n\n"
                "CLAW consists of three variables which control the maximum "
                "chroma point, the ratio of compression of the maximum "
                "chroma point, and exponent rate of compression.\n"
                "For wildly out of gamut imagery, the maximum chroma should "
                "be scaled downward.\n\n"
                "Degree of compression is the percentage of maximum chroma to "
                "compress the output down to.\n\n"
                "Rate of compression controls "
                "the shape of the compression curve, with a higher value "
                "corresponding to a less gradual and more agressive change",
            )

        CLAW_enable = streamlit.radio(
            "CLAW Gamut Compression", ["Enable", "Disable"], index=1
        )
        CLAW_maximum = streamlit.slider(
            label="CLAW Maximum",
            key="CLAW Rate",
            min_value=0.01,
            max_value=1.0,
            value=1.0,
            step=0.001,
        )
        CLAW_degree = streamlit.slider(
            label="CLAW Compression Degree",
            key="CLAW Degree",
            min_value=0.01,
            max_value=3.0 * (1.0 / 3.0),
            value=0.55,
            step=0.001,
        )
        CLAW_rate = streamlit.slider(
            label="CLAW Compression Rate",
            key="CLAW Rate",
            min_value=0.01,
            max_value=10.0,
            value=5.0,
            step=0.001,
        )

    region_1_1, region_1_2, region_1_3 = streamlit.beta_columns((2, 3, 2))
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
            max_value=25.0,
            value=10.0,
            step=0.25,
        )

    LUT = utilities.generic_aesthetic_transfer_function()
    LUT.set_transfer_details(
        contrast=contrast,
        middle_grey_in=middle_grey_input,
        middle_grey_out=middle_grey_output,
        ev_above_middle_grey=maximum_ev,
    )

    with region_1_2:
        # x_axis = "Open Domain (Log2 Scale)"
        # y_axis = "Closed Domain (Log2 Scale)"
        # plot_data = pandas.DataFrame({x_axis: LUT._linear, y_axis: LUT._LUT.table})
        # # plot_data = plot_data.set_index("Open Domain")
        # chart = (
        #     altair.Chart(plot_data)
        #     .transform_filter((altair.datum[x_axis] > 0) & (altair.datum[y_axis] > 0))
        #     .mark_line()
        #     .encode(
        #         x=altair.X(x_axis + ":Q", scale=altair.Scale(type="log", base=2)),
        #         y=altair.Y(y_axis + ":Q", scale=altair.Scale(type="log", base=2)),
        #     )
        # )

        # streamlit.altair_chart(chart, use_container_width=True)
        # streamlit.line_chart(data=plot_data, height=200)

        upload_image_help = streamlit.beta_expander("Custom Upload Image")
        with upload_image_help:
            streamlit.text_area(
                label="",
                key="Custom Upload Image Help",
                value="Choose a custom upload image for evaluation. Accepted file "
                "types are .EXR and .HDR.",
            )
        upload_image = streamlit.file_uploader(
            label="", key="Custom Upload Image", type=[".exr", ".hdr"]
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
                "Nightclub BT.709",
                "Red Xmas BT.709",
                "Blue Bar BT.709",
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
        cctf_help = streamlit.beta_expander("Display Hardware CCTF")
        with cctf_help:
            streamlit.text_area(
                label="",
                key="CCTF Help",
                value="The display hardware Electro Optical Transfer Function. "
                "This is a technical control that must match the hardware "
                "transfer function of the display used to view this web "
                "page. For most conventional displays such as sRGB-like or "
                "Display P3-like displays, this value should be 2.2. For "
                "broadcast BT.1886 displays, it should be 2.4. Some web "
                "browsers may be using colour management or inconsistent and / "
                "or broken software paths.",
            )
        cctf_encoding = streamlit.selectbox(
            label="",
            key="Display Hardware CCTF",
            options=utilities.enumerate_cctf_encodings(),
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
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

        luminance_weights_help = streamlit.beta_expander(
            "Creative Target Luminance Weights"
        )
        with luminance_weights_help:
            streamlit.text_area(
                label="",
                key="Luminance Weights Help",
                value="This is the chosen set of weights to use for the "
                "greyscale target. From a strictly light-meter centric "
                "vantage, the luminance weights are fixed based on the working "
                "space luminance weights of the primaries. However, given that "
                "the luminous efficacy function does not account for the "
                "perceptual ramifications of highly chrominous mixtures and "
                "the contribution to overall stimulated luminance-"
                "like sensation, the results are cognitively dissonant. LICH "
                "can use any luminance as a target, and will calculate the "
                "appropriate result accordingly. This selection box provides "
                "a set of roughly HK-like balanced weights versus the "
                "authoritative and strictly derived from luminous efficiacy "
                "function weights for comparison.",
            )
        luminance_weights = streamlit.selectbox(
            label="", key="Lumiance Weighting", options=["HK BT.709 weighting", "sRGB"]
        )

    if upload_image is None:
        default_image_path = helpers.get_dependency_local_path(default_image_path)
        default_image = colour.io.read_image_Imageio(default_image_path)[..., 0:3]
        reduced_image = default_image[::image_scale, ::image_scale, ...]
        original_image = default_image
    else:
        original_image = colour.io.read_image_Imageio(upload_image.read())[..., 0:3]
        reduced_image = original_image[::image_scale, ::image_scale, ...]

    img = utilities.adjust_exposure(reduced_image, exposure_adjustment)

    img_default = LUT.evaluate(np.clip(img, 0.0, None))
    img_default_final = utilities.apply_cctf_encoding(img_default, cctf_encoding)

    img_luminance_mapped = LUT.evaluate(
        utilities.calculate_luminance(img, luminance_weights)
    )

    img_luminance_mapped_final = utilities.apply_cctf_encoding(
        img_luminance_mapped, cctf_encoding
    )

    img_EVILS_LICH_render = utilities.calculate_EVILS_LICH(
        img, LUT.evaluate(utilities.calculate_luminance(img, luminance_weights))
    )
    img_EVILS_LICH_render_final = utilities.apply_cctf_encoding(
        np.clip(img_EVILS_LICH_render, 0.0, None), cctf_encoding
    )

    img_EVILS_CLAW_render = img
    if CLAW_enable is "Enable":
        img_EVILS_CLAW_render = utilities.calculate_EVILS_LICH(
            utilities.calculate_EVILS_CLAW(
                RGB_input=img,
                CLAW_compression=CLAW_rate,
                CLAW_identity_limit=None,
                CLAW_maximum_input=CLAW_maximum,
                CLAW_maximum_output=np.clip(1.0 - CLAW_degree, 0.01, 1.0),
            ),
            LUT.evaluate(utilities.calculate_luminance(img, luminance_weights)),
        )
    else:
        img_EVILS_CLAW_render = img_EVILS_LICH_render

    img_EVILS_CLAW_render_final = utilities.apply_cctf_encoding(
        img_EVILS_CLAW_render, cctf_encoding
    )

    with image_region_1_1:
        streamlit.image(
            img_default_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="Default Image Processing",
        )

        streamlit.image(
            img_EVILS_LICH_render_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="EVILS LICH Render (" + luminance_weights + " target)",
        )

    with image_region_1_2:
        streamlit.image(
            img_luminance_mapped_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="EVILS LICH Render, Luminance (" + luminance_weights + " target)",
        )

        streamlit.image(
            img_EVILS_CLAW_render_final,
            clamp=[0.0, 1.0],
            use_column_width=True,
            caption="EVILS CLAW Render (" + luminance_weights + " target)",
        )
