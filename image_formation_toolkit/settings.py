from pathlib import Path
import logging
import yaml
from pathlib import Path
from image_formation_toolkit._vendor.munch import Munch
import PyOpenColorIO as ocio

__app__ = "Experimental Image Formation Toolset"
__author__ = "THE HERMETIC BROTHERHOOD OV SPECTRA"
__license__ = "GPL3"
__version__ = "0.1.7"

__all__ = ["logger", "config", "LOCAL_DATA", "OCIO_VERSION", "EXTERNAL_DEPENDENCIES"]

config = Munch.fromDict(
    yaml.safe_load(open(str(Path(__file__).parent.parent / "config.yaml")))
)

logger = logging.getLogger(__app__)
logger.setLevel(logging._nameToLevel[config.logging.level])

LOCAL_DATA = Path.cwd() / "data"
OCIO_VERSION = ocio.__version__

EXTERNAL_DEPENDENCIES = {
    "ACES-1.2 Config": {
        "url": "https://github.com/colour-science/OpenColorIO-Configs"
        "/releases/download/v1.2/OpenColorIO-Config-ACES-1.2.zip",
        "size": 130123781,
        "family": "ocio-archive",
        "tags": ["ACES2065-1", "ACES"],
    },
    "Marcie ACES2065-1": {
        "filename": "DigitalLAD.2048x1556.exr",
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour"
        "/images/DigitalLAD.2048x1556.exr",
        "size": 25518832,
        "family": "image",
        "tags": ["test", "plate", "skin"],
        "metadata": {"encoding": "ACES2065-1"},
    },
    "CLF Test Image": {
        "filename": "CLF_testImagePrototype_v006.exr",
        "url": "https://raw.githubusercontent.com/alexfry/CLFTestImage"
        "/master/images/CLF_testImagePrototype_v006.exr",
        "size": 201549,
        "family": "image",
        "tags": ["test", "synthetic"],
        "metadata": {
            "encoding": "ACES2065-1",
        },
    },
    "Marcie 4K": {
        "filename": "marcie-4k.exr",
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour"
        "/images/marcie-4k.exr",
        "size": 63015668,
        "family": "image",
        "tags": ["test", "film", "skin"],
        "metadata": {
            "encoding": None,
        },
    },
    "Multi-Swatch Test Chart": {
        "filename": "syntheticChart_rec709.01.exr",
        "url": "https://raw.githubusercontent.com/sobotka/Testing_Imagery"
        "/master/AAD_BT_709_CAT_02_5600k_to_D65.exr",
        "size": 6690396,
        "family": "image",
        "tags": ["test", "CC24", "plots"],
        "metadata": {
            "encoding": "Linear-Rec709",
        },
    },
    "Alexa Models and Charts": {
        "filename": "syntheticChart_rec709.01.exr",
        "url": "https://raw.githubusercontent.com/sobotka/Testing_Imagery"
        "/master/alexa_BT_709.exr",
        "size": 18464820,
        "family": "image",
        "tags": ["test", "alexa"],
        "metadata": {
            "encoding": "Linear-Rec709",
        },
    },
    "Synthetic Testing Chart": {
        "filename": "syntheticChart_rec709.01.exr",
        "url": "https://raw.githubusercontent.com/sobotka/Testing_Imagery"
        "/master/syntheticChart_rec709.01.exr",
        "size": 2968371,
        "family": "image",
        "tags": ["test", "synthetic"],
        "metadata": {
            "encoding": "Linear-Rec709",
        },
    },
    "CC24 Chart, Synthetic": {
        "url": "https://raw.githubusercontent.com/sobotka/Testing_Imagery"
        "/master/CC24_BT-709_linear_1931.exr",
        "size": 12979565,
        "family": "image",
        "tags": ["test", "synthetic", "CC24"],
        "metadata": {
            "encoding": "Linear-Rec709",
        },
    },
    "CC24 Chart Photo": {
        "url": "https://raw.githubusercontent.com/sobotka/Testing_Imagery"
        "/master/CC24_d65adapted_bt709.exr",
        "size": 8927374,
        "family": "image",
        "tags": ["test", "video", "CC24"],
        "metadata": {
            "encoding": "Linear-Rec709",
        },
    },
    "CC24 Chart Photo, Cropped": {
        "filename": "syntheticChart_rec709.01.exr",
        "url": "https://raw.githubusercontent.com/sobotka/Testing_Imagery"
        "/master/CC24_d65adapted_bt709_cropped.exr",
        "size": 6278551,
        "family": "image",
        "tags": ["test", "video", "CC24"],
        "metadata": {
            "encoding": "Linear-Rec709",
        },
    },
    "Okja": {
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour"
        "/images/GM_TEST_002.0000.exr",
        "family": "image",
        "size": 8158434,
        "metadata": {"encoding": "ACES2065-1"},
    },
    "Products": {
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour"
        "/images/GM_TEST_001.0000.exr",
        "family": "image",
        "size": 5472858,
        "metadata": {"encoding": "ACES2065-1"},
    },
    "Red Xmas": {
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour"
        "/images/GM_TEST_011.0000.exr",
        "family": "image",
        "size": 9583965,
        "metadata": {"encoding": "ACES2065-1"},
    },
    "Red Xmas BT.709": {
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour"
        "/images/red_xmas_rec709.exr",
        "family": "image",
        "size": 8992075,
        "metadata": {"encoding": "Linear-Rec709"},
    },
    "Blue Bar": {
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour"
        "/images/GM_TEST_010.0187.exr",
        "family": "image",
        "size": 7267792,
        "metadata": {"encoding": "ACES2065-1"},
    },
    "Blue Bar BT.709": {
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour"
        "/images/blue_bar_709.exr",
        "family": "image",
        "size": 8896298,
        "metadata": {"encoding": "Linear-Rec709"},
    },
    "Nightclub BT.709": {
        "filename": "Matas_Alexa_Mini_sample_BT709.exr",
        "url": "https://raw.githubusercontent.com/sobotka/Testing_Imagery"
        "/master/Matas_Alexa_Mini_sample_BT709.exr",
        "family": "image",
        "size": 4481368,
        "metadata": {"encoding": "Linear-Rec709"},
    },
    "Brejon Lightsabers BT.709": {
        "filename": "mery_lightSaber_lin_srgb.exr",
        "url": "https://raw.githubusercontent.com/sobotka/Testing_Imagery"
        "/master/mery_lightSaber_lin_srgb.exr",
        "family": "image",
        "size": 4539254,
        "metadata": {"encoding": "Linear-Rec709"},
    },
    "Club DJ": {
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour"
        "/images/GM_TEST_004.0357.exr",
        "family": "image",
        "size": 6894347,
        "metadata": {"encoding": "ACES2065-1"},
    },
    "OCIO v2.0.0beta2": {
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour"
        "/data/ocio_streamlit_v2.0.0beta2.tar",
        "size": 28119040,
        "family": "lib",
        "metadata": {
            "prefix": "/home/appuser",
        },
    },
    "OCIO v2.0.0rc1": {
        "family": "lib",
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour"
        "/data/ocio_streamlit_v2.0.0rc1.tar",
        "metadata": {
            "prefix": "/home/appuser",
        },
    },
    "OCIO v2.0.0": {
        "family": "lib",
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour"
        "/data/ocio_streamlit_v2.0.0.tar",
        "metadata": {
            "prefix": "/home/appuser",
        },
    },
}
