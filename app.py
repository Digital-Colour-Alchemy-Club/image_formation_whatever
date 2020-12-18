import sys
from pathlib import Path
import streamlit as st
import numpy as np
import colour
from boltons.ecoutils import get_profile
import attr
from util import RemoteData

__author__ = "dcac@deadlythings.com"
__license__ = "GPL3"
__version__ = "0.1.3"

VERSION = '.'.join(__version__.split('.')[:2])
LOCAL_DATA = Path.cwd() / "data"
st.title("Image Formation Whatever")





@st.cache
def localize_dependencies(local_dir=LOCAL_DATA):
    data = EXTERNAL_DEPENDENCIES.copy()
    for name, remote_file in EXTERNAL_DEPENDENCIES.items():
        remote_file.download(output_dir=local_dir)
        data[name] = local_dir / remote_file.filename
    return data

@st.cache
def get_dependency(key, local_dir=LOCAL_DATA):
    remote_file = EXTERNAL_DEPENDENCIES[key]
    remote_file.download(output_dir=local_dir)
    return local_dir / remote_file.filename



def install_opencolorio():
    import gdown
    from plumbum.cmd import unzip
    url = "https://drive.google.com/uc?export=download&id=1sqgQ6e_aLffGiW-92XTRO7WYhLywaXh1"
    archive = gdown.cached_download(url, path=LOCAL_DATA/'OpenColorIO.zip')
    unzip['-d', '/'](archive)

def bootstrap():
    import imageio
    imageio.plugins.freeimage.download()
    _ = localize_dependencies()


def marcie():
    img_path = get_dependency("Marcie ACES2065-1")
    DIGITAL_LAD_ACES2065_1 = colour.read_image(img_path)[..., 0:3]
    DIGITAL_LAD_SRGB = colour.RGB_to_RGB(
        DIGITAL_LAD_ACES2065_1,
        colour.models.RGB_COLOURSPACE_ACES2065_1,
        colour.models.RGB_COLOURSPACE_sRGB,
        chromatic_adaptation_transform="Bradford"
    )
    st.image(DIGITAL_LAD_SRGB)







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


    # system info
    st.write("Streamlit instance info")
    st.write(get_profile())

    # colour-science info
    st.write("`colour-science` library info")
    st.write(colour.utilities.describe_environment())

    # locally-installed libraries
    st.write("Locally-installed libraries")
    library_versions = get_library_versions()
    for library, version in library_versions.keys():
        st.write(f"{library}: {version}")



def draw_main_page():
    st.write("""
    # Hello
    """)
    #bootstrap()
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
pages.insert(0, "Welcome.")

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
        url="https://drive.google.com/uc?id=1GrltrT4cb8PPhVIMII4fWRgIgAPdrpi",
        size=2551883,
    ),
    "CLF Test Image": RemoteData(
        filename="CLF_testImagePrototype_v006.exr",
        url="https://github.com/alexfry/CLFTestImage/blob/master/images/CLF_testImagePrototype_v006.exr",
        size=201549,
    )
}



# Draw main page
if selected_demo in demo_pages:
    demo_pages[selected_demo]()
else:
    draw_main_page()





