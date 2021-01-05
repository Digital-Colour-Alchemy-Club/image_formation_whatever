from functools import partial

import streamlit as st
import numpy as np

__app__ = "Experimental Image Formation Toolset"
__author__ = "THE HERMETIC BROTHERHOOD OV SPECTRA"
__license__ = "GPL3"
__version__ = "0.1.6"

st.set_page_config(page_title=__app__, layout="wide")


from bootstrap import run_bootstrap

run_bootstrap()

from helpers import st_stdout
from apps.diagnostics import diagnostics
from settings import LOCAL_DATA
from apps import image_formation


def ocio_skeleton_config():
    import PyOpenColorIO as ocio
    import ocioutils as ocu
    from helpers import get_dependency
    from operators import AestheticTransferFunction
    from colour import read_image

    key = st.selectbox(
        label="Test Image",
        options=[
            "Marcie 4K",
            "CLF Test Image",
        ],
    )

    # @st.cache
    def get_test_image(proxy=True):
        path = get_dependency(key)
        im = read_image(path)
        # if proxy:
        #     proxy_res = (1920, 1080)
        #     return (
        #         Image.fromarray(img)
        #             .thumbnail(size=proxy_res, resample=Image.BICUBIC)
        #     )
        return im

    def create_ocio_config(config=None):
        domain = np.array([-10, 15])
        logarithmic_shaper = ocio.NamedTransform(
            name="Logarithmic Shaper",
            forwardTransform=ocio.AllocationTransform(
                vars=np.log2(0.18 * np.power(2.0, domain))
            ),
        )
        image_formation_transform = ocio.ViewTransform(
            name="Image Formation Transform",
            fromReference=ocio.GroupTransform(
                [
                    ocio.FileTransform(
                        src="AestheticTransferFunction.csp",
                        interpolation=ocio.INTERP_TETRAHEDRAL,
                    ),
                ]
            ),
        )

        named_transforms = [
            logarithmic_shaper,
        ]

        view_transforms = [
            image_formation_transform,
        ]

        cfg = config or ocu.baby_config()

        for vt in view_transforms:
            cfg.addViewTransform(vt)

        for nt in named_transforms:
            cfg.addNamedTransform(nt)

        return cfg

    st.cache(get_test_image)
    st.cache(ocu.ocio_viewer, max_entries=10)

    cfg = create_ocio_config()

    with st.sidebar:
        source = st.selectbox(label="Image Encoding", options=cfg.getColorSpaceNames())
        display = st.selectbox(
            label="Display",
            options=cfg.getActiveDisplays() or cfg.getDisplays(),
        )
        view = st.selectbox(label="View", options=cfg.getViews(display))
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

    image = ocu.ocio_viewer(
        image,
        source=source,
        display=display,
        view=view,
        exposure=exposure,
        contrast=contrast,
        gamma=gamma,
        style=style,
        config=cfg,
    )

    image_placeholder.image(image, clamp=[0, 1], use_column_width=True)

    with st_stdout("code"):
        print(cfg)


demo_pages = {
    "Experimental Image Formation": image_formation.application_experimental_image_formation,
    "Diagnostics": partial(diagnostics, LOCAL_DATA),
    "Baby bones": ocio_skeleton_config,
}

# Draw sidebar
pages = list(demo_pages.keys())

applications = st.sidebar.selectbox(
    "Applications Version {}".format(__version__), pages
)


demo_pages[applications]()
