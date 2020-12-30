from pathlib import Path
import streamlit as st
import numpy as np
import colour
from boltons.ecoutils import get_profile
from util import RemoteData, build_ocio
from lookup_methods import (
    norm_lookup,
    generic_aesthetic_transfer_function
)

import logging

st.set_page_config(
    page_title="Experimental Image Formation Toolset",
    layout="wide")

__app__ = "Image Formation Whatever"
__author__ = "dcac@deadlythings.com"
__license__ = "GPL3"
__version__ = "0.1.4"

VERSION = '.'.join(__version__.split('.')[:2])
LOCAL_DATA = Path.cwd() / "data"

logger = logging.getLogger(__app__)


def get_dependency(key, local_dir=LOCAL_DATA):
    remote_file = EXTERNAL_DEPENDENCIES[key]
    remote_file.download(output_dir=local_dir)
    return local_dir / remote_file.filename


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


def experimental_image_formation():
    # @st.cache
    LUT = generic_aesthetic_transfer_function()

    col1, col2 = st.beta_columns([1, 3])
    with col1:
        EOTF = st.number_input(
            label="Display Hardware EOTF",
            min_value=1.0,
            max_value=3.0,
            value=2.2,
            step=0.01)
        exposure = st.slider(
            label="Exposure Adjustment",
            min_value=-10.0,
            max_value=+10.0,
            value=0.0,
            step=0.25)
        contrast = st.slider(
            label="Contrast",
            min_value=0.01,
            max_value=3.00,
            value=1.0,
            step=0.01
        )
        shoulder_contrast = st.slider(
            label="Shoulder Contrast",
            min_value=0.01,
            max_value=1.00,
            value=1.0,
            step=0.01
        )
        gamut_clipping = st.checkbox("Gamut Clipping Indicator")

    def apply_inverse_EOTF(RGB):
        return np.ma.power(RGB, (1.0 / EOTF)).filled(fill_value=0.0)

    # @st.cache
    def video_buffer(x):
        return ((2.0**exposure) * x)

    # @st.cache
    @norm_lookup(degree=5, weights=[1.22, 1.20, 0.58])
    def video_buffer_norm(x):
        return video_buffer(x)

    @st.cache
    def get_marcie():
        img_path = get_dependency("Marcie 4K")
        img = colour.read_image(img_path)[..., 0:3]
        return img

    # st.subheader('maxRGB')
    with col2:
        LUT.set_transfer_details(
            contrast=contrast,
            shoulder_contrast=shoulder_contrast
        )
        st.line_chart(data=LUT._LUT.table)

        img = LUT.apply_maxRGB(
            video_buffer(get_marcie()), gamut_clipping)
        st.image(
            apply_inverse_EOTF(img),
            clamp=[0., 1.],
            use_column_width=True,
            caption=LUT._LUT.name)

        img = LUT.apply_per_channel(
            video_buffer(get_marcie()), gamut_clipping)
        st.image(
            apply_inverse_EOTF(img),
            clamp=[0., 1.],
            use_column_width=True,
            caption="Generic Per Channel")

    # st.subheader('Yellow-Weighted Norm')
    # img = apply_inverse_EOTF(video_buffer_norm(get_marcie()))
    # st.image(img, clamp=[0., 1.], use_column_width=True)


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
    pass
    # bootstrap()


demo_pages = {
    "Experimental Image Formation": experimental_image_formation,
    "Diagnostics": diagnostics,
    "Installation Tools": installation_tools
}

# Draw sidebar
pages = list(demo_pages.keys())

applications = st.sidebar.selectbox(
    "Applications Version {}".format(VERSION),
    pages
)


EXTERNAL_DEPENDENCIES = {
    # "ACES-1.2 Config": RemoteData(
    #     filename="OpenColorIO-Config-ACES-1.2.zip",
    #     url="https://github.com/colour-science/OpenColorIO-Configs/"
    #       "releases/download/v1.2/OpenColorIO-Config-ACES-1.2.zip",
    #     size=130123781,
    # ),
    "Marcie ACES2065-1": RemoteData(
        filename="DigitalLAD.2048x1556.exr",
        url="https://zozobra.s3.us-east-1.amazonaws.com/colour/"
            "images/DigitalLAD.2048x1556.exr",
        size=25518832,
    ),
    "CLF Test Image": RemoteData(
        filename="CLF_testImagePrototype_v006.exr",
        url="https://raw.githubusercontent.com/alexfry/CLFTestImage/"
            "master/images/CLF_testImagePrototype_v006.exr",
        size=201549,
    ),
    "Marcie 4K": RemoteData(
        filename="marcie-4k.exr",
        url="https://zozobra.s3.us-east-1.amazonaws.com/colour/"
            "images/marcie-4k.exr",
        size=63015668,
    ),
}


# Draw main page
if applications in demo_pages:
    demo_pages[applications]()
else:
    installation_tools()
