from functools import partial

import streamlit as st

__app__ = "Experimental Image Formation Toolset"
__author__ = "THE HERMETIC BROTHERHOOD OV SPECTRA"
__license__ = "GPL3"
__version__ = "0.1.6"

st.set_page_config(page_title=__app__, layout="wide")

from bootstrap import run_bootstrap

# download the openexr plugin for imageio
run_bootstrap()

from apps.ocio_config_generator import ocio_skeleton_config
from apps.diagnostics import diagnostics
from settings import LOCAL_DATA
from apps import (
    image_formation00,
    image_formation01,
    image_formation02,
    ocio_formation00,
)


demo_pages = {
    "EVILS v0.1": image_formation02.application_image_formation_02,
    "Luminance Evaluation": image_formation01.application_image_formation_01,
    "Experimental Image Formation": image_formation00.application_experimental_image_formation_00,
    "Diagnostics": partial(diagnostics, LOCAL_DATA),
    "OpenColorIO Formation": ocio_formation00.application_ocio_formation_00,
}

# Draw sidebar
pages = list(demo_pages.keys())

applications = st.sidebar.selectbox(
    "Applications Version {}".format(__version__), pages
)


demo_pages[applications]()
