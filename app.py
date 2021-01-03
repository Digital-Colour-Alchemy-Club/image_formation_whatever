import logging
from pathlib import Path
import sys

from boltons.ecoutils import get_profile
import colour
import fs
import streamlit as st
import numpy as np

from util import st_stdout
from ocioutils import build_ocio, fetch_ocio
from data_utilities import st_file_downloader
import image_formation

__app__ = "Experimental Image Formation Toolset"
__author__ = "THE HERMETIC BROTHERHOOD OV SPECTRA"
__license__ = "GPL3"
__version__ = "0.1.5"

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
                        fetch_ocio(prefix=prefix, version=version, force=force)
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

    setup_opencolorio(prefix='/home/appuser', version="2.0.0beta2", force=False)


def ocio_skeleton_config():
    import PyOpenColorIO as ocio
    import ocioutils as ocu
    from PIL import Image
    key = st.selectbox(
            label="Test Image",
            options=[
                'Marcie 4K',
                'CLF Test Image',
            ]
        )
    def get_test_image(proxy=True):
        from data_utilities import get_dependency
        from colour import read_image

        st.cache(func=read_image)
        img = read_image(get_dependency(key))

        # if proxy:
        #     proxy_res = (1920, 1080)
        #     return (
        #         Image.fromarray(img)
        #             .thumbnail(size=proxy_res, resample=Image.BICUBIC)
        #     )
        return img

    def create_ocio_config(config=None):
        domain = np.array([-10, 15])

        logarithmic_shaper = ocio.NamedTransform(
            name='Logarithmic Shaper',
            forwardTransform=ocio.AllocationTransform(
                vars=np.log2(0.18 * np.power(2., domain))
            )
        )

        aesthetic_transfer_function = ocio.NamedTransform(
            name='Aesthetic Transfer Function',
            forwardTransform=ocio.FileTransform(
                src="AestheticTransferFunction.csp",
                interpolation=ocio.INTERP_TETRAHEDRAL,
            ),
        )

        image_formation_transform = (
            ocio.ViewTransform(
                name='Image Formation Transform',
                fromReference=ocio.GroupTransform([
                    ocio.FileTransform(
                        src="AestheticTransferFunction.csp",
                        interpolation=ocio.INTERP_TETRAHEDRAL,
                    ),
                ])
            )
        )

        named_transforms = [
            logarithmic_shaper,
            aesthetic_transfer_function,
        ]

        view_transforms = [
            image_formation_transform,
        ]

        cfg = config or ocu.baby_config()

        for vt in view_transforms:
            cfg.addViewTransform(vt)

        for nt in named_transforms:
            cfg.addNamedTransform(nt)

        return cfg

    exposure_contrast_transform = ocio.ExposureContrastTransform(
        exposure=st.slider(
            label='Exposure Adjustment',
            min_value=-10.0,
            max_value=+10.0,
            value=0.0,
            step=0.25,
        ),
        contrast=st.slider(
            label="Contrast",
            min_value=0.01,
            max_value=3.00,
            value=1.00,
            step=0.1
        ),
        gamma=st.slider(
            label="Gamma",
            min_value=0.2,
            max_value=4.00,
            value=1.0,
            step=0.1,
        ),
        style=ocio.EXPOSURE_CONTRAST_LINEAR
    )

    cfg = create_ocio_config()

    proc = cfg.getProcessor(
        ocio.GroupTransform([
            exposure_contrast_transform,
            ocio.ColorSpaceTransform(
                'linear', 'sRGB'
            )
        ])
    )

    img = np.copy(get_test_image())
    st.cache(func=ocu.apply_ocio_processor)
    st.image(
        ocu.apply_ocio_processor(proc, img),
        clamp=[0, 1]
    )

    with st_stdout("code"):
        print(cfg)






demo_pages = {
    "Experimental Image Formation":
        image_formation.application_experimental_image_formation,
    "Diagnostics":
        diagnostics,
    "Installation Tools":
        installation_tools,
    "Baby bones":
        ocio_skeleton_config
}

# Draw sidebar
pages = list(demo_pages.keys())

applications = st.sidebar.selectbox(
    "Applications Version {}".format(__version__),
    pages
)

installation_tools()
demo_pages[applications]()

