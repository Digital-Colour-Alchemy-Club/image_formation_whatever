from functools import partial

import streamlit as st

from image_formation_apps.diagnostics import diagnostics
from image_formation_toolkit.settings import LOCAL_DATA
from image_formation_apps import (
    image_formation00,
    image_formation01,
    image_formation02,
    ocio_formation00,
)

__app__ = "Experimental Image Formation Toolset"
__author__ = "THE HERMETIC BROTHERHOOD OV SPECTRA"
__license__ = "GPL3"
__version__ = "0.1.6"

st.set_page_config(page_title=__app__, layout="wide")


def bootstrap_imageio():
    if not st.session_state.get("imageio_initialized", False):
        try:
            import imageio

            imageio.plugins.freeimage.download()
            st.session_set["imageio_initialized"] = True

        except ImportError:
            pass


demo_pages = {
    "EVILS v0.1": image_formation02.application_image_formation_02,
    "Luminance Evaluation": image_formation01.application_image_formation_01,
    "Experimental Image Formation": image_formation00.application_experimental_image_formation_00,
    "Diagnostics": partial(diagnostics, LOCAL_DATA),
    "OpenColorIO Formation": ocio_formation00.application_ocio_formation_00,
}

# Install imageio freeimage plugin (i.e., for EXR support)
bootstrap_imageio()

# Draw sidebar
pages = list(demo_pages.keys())

applications = st.sidebar.selectbox(
    "Applications Version {}".format(__version__), pages
)


demo_pages[applications]()
