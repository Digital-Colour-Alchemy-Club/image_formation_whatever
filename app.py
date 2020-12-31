import logging
from pathlib import Path
import sys

from boltons.ecoutils import get_profile
import colour
import fs
import streamlit as st

from util import build_ocio, st_stdout
from data_utilities import EXTERNAL_DEPENDENCIES, st_file_downloader, get_dependency
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

    # Build and install OCIO
    if build_libs:
        build_ocio()


def diagnostics():
    st.header("Streamlit instance info")
    st.write(get_profile())
    st.header("Python")
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
    with st_stdout("code"):
        fs.open_fs(str(LOCAL_DATA)).tree()
    st.write(LOCAL_DATA)


def installation_tools():
    bootstrap(build_libs=False)
    try:
        import PyOpenColorIO as ocio
    except ImportError:
        class Null(object):
            def __getattr__(self, name): return None
            def __bool__(self): return False
        ocio = Null()
        
    def fetch_and_install_prebuilt_ocio(prefix="/home/appuser",
                                        version="2.0.0beta2",
                                        force=False):
        if not force and version == ocio.__version__:
            return

        # TODO: catch exception, invoke build_ocio if resource not available
        archive_path = get_dependency(f"OCIO v{version}", local_dir=LOCAL_DATA)
        archive = fs.open_fs(f"tar://{archive_path}")

        with fs.open_fs(prefix, writeable=True) as dst:
            fs.copy.copy_dir(archive, prefix, dst, '.')
        dst = fs.open_fs(prefix)
        sys.path.extend(
            [dst.getsyspath(p.path) for p in dst.glob("**/site-packages/")])

        try:
            import PyOpenColorIO
        except ImportError:
            with st_stdout("error"):
                print("""***NICE WORK, GENIUS: ***
                You've managed to build, archive, retrieve, and deploy OCIO...
                yet you couldn't manage to import PyOpenColorIO.""")
        logger.debug(f"OpenColorIO v{version} installed!")

    # Offer archive of existing libraries
    if ocio:
        with st_stdout('write'):
            print(f"OCIO Library path: {ocio.__file__}")

        lib_archive = Path(ocio.__file__).parents[3] / "ocio_streamlit.tar"

        if lib_archive.exists():
            # archive generated at build time (see `build_ocio` method)
            st_file_downloader(lib_archive, f"OCIO v{ocio.__version__} libs")

    fetch_and_install_prebuilt_ocio(version="2.0.0beta2", force=False)


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

