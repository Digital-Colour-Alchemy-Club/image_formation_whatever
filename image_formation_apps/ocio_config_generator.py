import streamlit as st

from image_formation_apps.helpers import st_stdout


# @st.cache(max_entries=10)
from image_formation_toolkit.ocioutils import add_aesthetic_transfer_function_to_config


def downsize_image(image, factor=3):
    assert 0 < factor < 11
    return image[::factor, ::factor, ...]


def ocio_skeleton_config():
    import PyOpenColorIO as ocio
    import image_formation_toolkit.ocioutils as ocu
    from image_formation_apps.helpers import get_dependency_local_path
    from image_formation_toolkit.operators import AestheticTransferFunction
    from colour import read_image

    def get_test_image():
        key = st.selectbox(
            label="Test Image",
            options=["Marcie 4K", "CLF Test Image", "Upload EXR"],
        )
        upload_image_placeholder = st.empty()
        image_scale = st.slider(
            label="Downsize Image",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

        if key == "Upload EXR":
            img = upload_image_placeholder.file_uploader(
                label="Input Image", type=[".exr"]
            )
        else:
            img = get_dependency_local_path(key)

        return downsize_image(
            read_image(img.read() if hasattr(img, "read") else img), factor=image_scale
        )

    def st_ocio_viewer(image, viewer_config):
        with st.sidebar:
            source = st.selectbox(
                label="Image Encoding", options=viewer_config.getColorSpaceNames()
            )
            display = st.selectbox(
                label="Display",
                options=viewer_config.getActiveDisplays()
                or viewer_config.getDisplays(),
            )
            view = st.selectbox(label="View", options=viewer_config.getViews(display))

            exposure = st.slider(
                label="Exposure Adjustment",
                min_value=-10.0,
                max_value=+10.0,
                value=0.0,
                step=0.25,
            )
            contrast = st.slider(
                label="Contrast", min_value=0.01, max_value=3.00, value=1.00, step=0.1
            )
            gamma = st.slider(
                label="Gamma",
                min_value=0.2,
                max_value=4.00,
                value=1.0,
                step=0.1,
            )
            style = st.selectbox(
                label="Exposure/Contrast Style",
                options=[
                    ocio.EXPOSURE_CONTRAST_LINEAR,
                    ocio.EXPOSURE_CONTRAST_LOGARITHMIC,
                    ocio.EXPOSURE_CONTRAST_VIDEO,
                ],
                format_func=ocio.ExposureContrastStyleToString,
            )

        return ocu.ocio_viewer(
            image,
            source=source,
            display=display,
            view=view,
            exposure=exposure,
            contrast=contrast,
            gamma=gamma,
            style=style,
            config=viewer_config,
        )

    image_placeholder = st.empty()

    aesthetic_transfer_function = AestheticTransferFunction(
        middle_grey_in=st.number_input(
            label="Middle Grey Input Value, Radiometric",
            min_value=0.01,
            max_value=1.0,
            value=0.18,
            step=0.001,
        ),
        middle_grey_out=st.number_input(
            label="Middle Grey Output Display Value, Radiometric",
            min_value=0.01,
            max_value=1.0,
            value=0.18,
            step=0.001,
        ),
        ev_above_middle_grey=st.slider(
            label="Maximum EV Above Middle Grey",
            min_value=1.0,
            max_value=15.0,
            value=4.0,
            step=0.25,
        ),
        # exposure = st.slider(
        #     label="Exposure Adjustment",
        #     min_value=-10.0,
        #     max_value=+10.0,
        #     value=0.0,
        #     step=0.25
        # ),
        contrast=st.slider(
            label="Contrast", min_value=0.01, max_value=3.00, value=1.75, step=0.01
        ),
        shoulder_contrast=st.slider(
            label="Shoulder Contrast",
            min_value=0.01,
            max_value=1.00,
            value=1.0,
            step=0.01,
        ),
        gamut_clip=st.checkbox(
            "Gamut Clip to Maximum",
            value=True,
        ),
        gamut_warning=st.checkbox("Exceeds Gamut Indicator"),
    )

    image = get_test_image()

    # temporary hack (cuz i'm tired) to get the image / contrast adjustment under the LUT
    image = aesthetic_transfer_function.apply(image)
    # eh... TODO...
    clf = aesthetic_transfer_function.generate_clf()
    range_shaper = clf[0]
    # todo -- implement clf.py...
    # colour.write_LUT(clf, 'AestheticTransferFunction.clf')

    viewer_config = ocu.baby_config()

    image_placeholder.image(
        st_ocio_viewer(image, viewer_config), clamp=[0, 1], use_column_width=True
    )

    cfg = add_aesthetic_transfer_function_to_config(
        aesthetic_transfer_function, config=viewer_config
    )

    with st_stdout("code"):
        print(cfg)
