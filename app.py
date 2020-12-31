import logging
from pathlib import Path
import sys

from boltons.ecoutils import get_profile
import colour
import fs
import streamlit as st

from util import st_stdout
from ocioutils import build_ocio, fetch_and_install_prebuilt_ocio
from data_utilities import st_file_downloader
import image_formation

__app__ = "Experimental Image Formation Toolset"
__author__ = "THE HERMETIC BROTHERHOOD OV SPECTRA"
__license__ = "GPL3"
__version__ = "0.1.4"

LOCAL_DATA = Path.cwd() / "data"

logger = logging.getLogger(__app__)

st.set_page_config(
    page_title=__app__,
    layout="wide")


def bootstrap(build_libs=False):
    # Install imageio freeimage plugin (i.e., for EXR support)
    import imageio
    imageio.plugins.freeimage.download()
    #
    # # Build and install OCIO
    # if build_libs:
    #     build_ocio()


def diagnostics():
    st.header("Streamlit instance info")
    st.write(get_profile())
    st.header("Python ")
    st.subheader("System paths")
    st.write(sys.path)
    st.subheader("`colour-science` library info")
    with st_stdout("code"):
        colour.utilities.describe_environment()
    st.subheader("Locally-installed libraries")
    try:
        import PyOpenColorIO as ocio
        st.write(f"PyOpenColorIO: v{ocio.__version__}")
    except ImportError:
        pass

    st.header("Local contents")
    st.write(LOCAL_DATA)
    with st_stdout("code"):
        fs.open_fs(str(LOCAL_DATA)).tree()



def installation_tools():
    bootstrap(build_libs=False)

    def setup_opencolorio(prefix='/usr/local', version="2.0.0beta2", force=False):
        try:
            import PyOpenColorIO as ocio
        except ImportError:
            class Null(object):
                def __getattr__(self, name): return None
                def __bool__(self): return False
            ocio = Null()

        def install_opencolorio(prefix=prefix, version=version, force=force):
            with st.spinner("Setting up OpenColorIO..."):
                if force:
                    build_ocio(prefix=prefix, version=version, force=force,
                               build_apps=True, build_shared=False)
                else:
                    try:
                        fetch_and_install_prebuilt_ocio(prefix=prefix, version=version, force=force)
                    except:
                        build_ocio(prefix=prefix, version=version, force=force,
                                   build_apps=True, build_shared=False)

        # Offer archive of existing libraries
        if ocio:
            lib_archive = Path(ocio.__file__).parents[3] / "ocio_streamlit.tar"

            if lib_archive.exists():
                # archive generated at build time (see `build_ocio` method)
                st_file_downloader(lib_archive, f"OCIO v{ocio.__version__} libs")

        install_opencolorio(prefix=prefix, version=version, force=False)

    setup_opencolorio(prefix='/usr/local', version="2.0.0beta3", force=True)


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

installation_tools()
demo_pages[applications]()

