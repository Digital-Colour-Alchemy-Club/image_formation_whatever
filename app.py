import sys
import subprocess
import pkg_resources

import logging

# The proceeding auto-pip installation process should streamline
# local Streamlit development. In theory, the OpenColorIO check
# might be handled in the missing component here?
required = {
    "colour-science",
    "boltons",
    "fs",
    "gdown",
    "plumbum",
    "numpy"
    # , "PyOpenColorIO"
}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    # Check if PyOpenColorIO is in the missing list, and if
    # it is, handle it here.
    # build_ocio(version='master')

    python = sys.executable
    subprocess.check_call(
        [
            python,
            "-m",
            "pip",
            "install",
            *missing
        ],
        stdout=subprocess.DEVNULL
    )

# The following will trip PEP errors, so explicitly turn off
# the errors via the noqa: E402 comment. There might be a more
# PEP compliant method to import after the top of file with the above
# logic, and it would be great to do it.

from pathlib import Path  # noqa: E402
import streamlit as st  # noqa: E402
import numpy as np  # noqa: E402
import colour  # noqa: E402
from boltons.ecoutils import get_profile  # noqa: E402
import attr  # noqa: E402
from util import RemoteData, build_ocio  # noqa: E402
from lookup_methods import (
    norm_lookup,
    maxrgb_lookup,
    generic_aesthetic_transfer_function   # noqa: E402
)
import fs  # noqa: E402
from fs.zipfs import ZipFS  # noqa: E402

st.set_page_config(layout="wide")

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
#     url = "https://drive.google.com/uc?export=download&id="
#           "1sqgQ6e_aLffGiW-92XTRO7WYhLywaXh1"
#     archive = gdown.cached_download(url, path=str(
#       LOCAL_DATA/'OpenColorIO.zip'), postprocess=extract)


def bootstrap():
    import imageio
    imageio.plugins.freeimage.download()
    _ = localize_dependencies()
    build_ocio()


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

    st.subheader("ACES Marcie")
    st.image(DIGITAL_LAD_ACES2065_1, clamp=[0., 1.])

    st.subheader("Not-ACES Marcie")
    img_path = get_dependency("Marcie 4K")
    img = colour.read_image(img_path)[..., 0:3]
    st.image(img, clamp=[0., 1.])


def lookup_method_tests():
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
    def LUT_buffer(x, LUT):
        return x

    # @st.cache
    # @maxrgb_lookup(LUT=get_LUT()._LUT)
    # def video_buffer_maxrgb(x):
    #     return video_buffer(x)

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
    'Marcie': marcie,
    'Lookup Methods': lookup_method_tests,
}

# Draw sidebar
pages = list(demo_pages.keys())
pages.insert(0, "Welcome")

st.sidebar.title(f"Demos v{VERSION}")
selected_demo = st.sidebar.radio("", pages)


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
if selected_demo in demo_pages:
    demo_pages[selected_demo]()
else:
    draw_main_page()
