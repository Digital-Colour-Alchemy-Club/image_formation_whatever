import logging
from pathlib import Path

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
    if build_libs:
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
    bootstrap(build_libs=False)

    try:
        import PyOpenColorIO as ocio
    except ImportError:
        class Null(object):
            def __getattr__(self, name): return None
            def __bool__(self): return False
        ocio = Null()
        
    def fetch_and_install_prebuilt_ocio(version="2.0.0beta2", force=False):
        status = st.empty()

        if not force and version == ocio.__version__:
            status.info(f"OCIO v{version} already installed!")
            return

        dst_path = Path('/')
        archive_path = get_dependency(f"OCIO v{version}", local_dir=LOCAL_DATA)
        archive = fs.open_fs(f"tar://{archive_path}")

        # copy archive contents to dst path
        def _update_status(src_fs, src_path, dst_fs, dst_path):
            status.info(f"Extracted {dst_path}...")

        with fs.open_fs(str(dst_path), writeable=True) as dst:
            fs.copy.copy_fs_if_newer(archive, dst, on_copy=_update_status)

        # add PyOpenColorIO to the system path
        pyocio_dir = dst_path / list(archive.glob("**/site-packages/"))[0].path

        if pyocio_dir.exists():
            import sys
            sys.path.append(pyocio_dir)
            status.success(f"OpenColorIO v{version} installed!")

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

