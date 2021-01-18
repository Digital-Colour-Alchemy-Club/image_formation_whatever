from pathlib import Path
import logging
import timber
import yaml
from munch import Munch

__app__ = "Experimental Image Formation Toolset"
__author__ = "THE HERMETIC BROTHERHOOD OV SPECTRA"
__license__ = "GPL3"
__version__ = "0.1.7"

__all__ = ["logger", "config", "LOCAL_DATA", "OCIO_VERSION", "EXTERNAL_DEPENDENCIES"]


config = Munch.fromDict(yaml.safe_load(open("config.yaml")))

logger = logging.getLogger(__app__)
logger.setLevel(logging._nameToLevel[config.logging.level])
timber_handler = timber.TimberHandler(
    config.services.timber.key,
    source_id=config.services.timber.source_id,
)
logger.addHandler(timber_handler)


LOCAL_DATA = Path.cwd() / "data"
OCIO_VERSION = config.libs.OpenColorIO.version

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
}
