from pathlib import Path
import streamlit as st
import numpy as np
import colour
from boltons.ecoutils import get_profile
from util import build_ocio
import image_formation

import logging

__app__ = "Experimental Image Formation Toolset"
__author__ = "THE HERMETIC BROTHERHOOD OV SPECTRA"
__license__ = "GPL3"
__version__ = "0.1.4"

LOCAL_DATA = Path.cwd() / "data"

logger = logging.getLogger(__app__)

st.set_page_config(
    page_title=__app__,
    layout="wide")

def bootstrap():
    def localize_dependencies(local_dir=LOCAL_DATA):
        data = EXTERNAL_DEPENDENCIES.copy()
        for name, remote_file in EXTERNAL_DEPENDENCIES.items():
            path = remote_file.download(output_dir=local_dir)
            data[name] = path
            assert(Path(path).exists())
        return data

    # Install imageio freeimage plugin (i.e., for EXR support)
    import imageio
    imageio.plugins.freeimage.download()

    # Download all app dependencies
    _ = localize_dependencies()
    build_ocio()

    # Build and install OCIO
    build_ocio()


def diagnostics():
    @st.cache
    def get_library_versions():
        libraries = {}

        try:
            import OpenImageIO as oiio
            libraries['OpenImageIO'] = oiio.__version__
        except ImportError:
            pass

        try:
            import PyOpenColorIO as ocio
            libraries['OpenColorIO'] = ocio.__version__
        except ImportError:
            pass

        return libraries

    st.write("### Streamlit instance info")
    st.write(get_profile())

    st.write("### `colour-science` library info")
    st.write(colour.utilities.describe_environment())

    st.write("### Locally-installed libraries")
    library_versions = get_library_versions()
    for library, version in library_versions.items():
        st.write(f"{library}: {version}")


def installation_tools():
    bootstrap()


demo_pages = {
    "Experimental Image Formation":
        image_formation.application_experimental_image_formation,
    "Diagnostics":
        diagnostics,
    "Installation Tools":
        installation_tools
}

# Draw sidebar
pages = list(demo_pages.keys())

applications = st.sidebar.selectbox(
    "Applications Version {}".format(__version__),
    pages
)


# Draw main page
if applications in demo_pages:
    demo_pages[applications]()
else:
    installation_tools()
