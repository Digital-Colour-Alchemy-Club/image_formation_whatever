import sys
from pathlib import Path
import streamlit as st
import numpy as np
import colour
from boltons.ecoutils import get_profile
import attr
from util import RemoteData, build_ocio
import fs
from fs.zipfs import ZipFS

import logging

__app__ = "Image Formation Whatever"
__author__ = "dcac@deadlythings.com"
__license__ = "GPL3"
__version__ = "0.1.3"

VERSION = '.'.join(__version__.split('.')[:2])
LOCAL_DATA = Path.cwd() / "data"
st.title(__app__)


logger = logging.getLogger(__app__)




def localize_dependencies(local_dir=LOCAL_DATA):
    data = EXTERNAL_DEPENDENCIES.copy()
    for name, remote_file in EXTERNAL_DEPENDENCIES.items():
        path = remote_file.download(output_dir=local_dir)
        data[name] = path
        assert(Path(path).exists())
    return data


def get_dependency(key, local_dir=LOCAL_DATA):
    remote_file = EXTERNAL_DEPENDENCIES[key]
    remote_file.download(output_dir=local_dir)
    return local_dir / remote_file.filename


def archive_ocio_payload(install_path='/home/appuser',
                         filename='ocio_streamlit.zip'):
    root = fs.open_fs(install_path)
    archive_path = f"{install_path}/{filename}"
    with ZipFS(f"{archive_path}", write=True) as archive:
        fs.copy.copy_dir(root, 'include', archive, install_path)
        fs.copy.copy_dir(root, 'lib', archive, install_path)
        logger.debug(f"Archived {archive_path}")
    return archive_path

# def install_opencolorio():
#     import gdown
#     from functools import partial
#     from plumbum import local
#     extract = partial(gdown.extractall, to='/home/appuser')
#     url = "https://drive.google.com/uc?export=download&id=1sqgQ6e_aLffGiW-92XTRO7WYhLywaXh1"
#     archive = gdown.cached_download(url, path=str(LOCAL_DATA/'OpenColorIO.zip'), postprocess=extract)


def bootstrap():
    import imageio
    imageio.plugins.freeimage.download()
    _ = localize_dependencies()
    build_ocio(version='master')



def marcie():
    img_path = get_dependency("Marcie ACES2065-1")
    DIGITAL_LAD_ACES2065_1 = colour.read_image(img_path)[..., 0:3]
    DIGITAL_LAD_SRGB = colour.RGB_to_RGB(
        DIGITAL_LAD_ACES2065_1,
        colour.models.RGB_COLOURSPACE_ACES2065_1,
        colour.models.RGB_COLOURSPACE_sRGB,
        chromatic_adaptation_transform="Bradford"
    )
    st.image(DIGITAL_LAD_SRGB, clamp=[0., 1.])







def about():
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

    st.subheader("Local contents")
    data_dir = fs.open_fs(str(LOCAL_DATA))
    data_dir.tree()



def draw_main_page():
    st.write("""
    # Hello
    """)
    bootstrap()
    st.info("""
        :point_left: **To get started, choose a demo on the left sidebar.**
    """)
    st.balloons()



demo_pages = {
    'About': about,
    'Marcie': marcie
}

# Draw sidebar
pages = list(demo_pages.keys())
pages.insert(0, "Welcome")

st.sidebar.title(f"Demos v{VERSION}")
selected_demo = st.sidebar.radio("", pages)


EXTERNAL_DEPENDENCIES = {
    "ACES-1.2 Config": RemoteData(
        filename="OpenColorIO-Config-ACES-1.2.zip",
        url="https://github.com/colour-science/OpenColorIO-Configs/releases/download/v1.2/OpenColorIO-Config-ACES-1.2.zip",
        size=130123781,
    ),
    "Marcie ACES2065-1": RemoteData(
        filename="DigitalLAD.2048x1556.exr",
        url="https://zozobra.s3.us-east-1.amazonaws.com/colour/images/DigitalLAD.2048x1556.exr",
        size=25518832,
    ),
    "CLF Test Image": RemoteData(
        filename="CLF_testImagePrototype_v006.exr",
        url="https://raw.githubusercontent.com/alexfry/CLFTestImage/master/images/CLF_testImagePrototype_v006.exr",
        size=201549,
    )
}



# Draw main page
if selected_demo in demo_pages:
    demo_pages[selected_demo]()
else:
    draw_main_page()





