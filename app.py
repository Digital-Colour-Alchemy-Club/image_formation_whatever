from functools import partial

import streamlit as st

__app__ = "Experimental Image Formation Toolset"
__author__ = "THE HERMETIC BROTHERHOOD OV SPECTRA"
__license__ = "GPL3"
__version__ = "0.1.6"

st.set_page_config(page_title=__app__, layout="wide")

from bootstrap import run_bootstrap

# Fetch and install PyOpenColorIO before importing anything else!
run_bootstrap()

from apps.ocio_config_generator import ocio_skeleton_config
from apps.diagnostics import diagnostics
from settings import LOCAL_DATA
from apps import image_formation

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
