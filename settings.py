from pathlib import Path
from logging import getLogger

__app__ = "Experimental Image Formation Toolset"
__author__ = "THE HERMETIC BROTHERHOOD OV SPECTRA"
__license__ = "GPL3"
__version__ = "0.1.6"

logger = getLogger(__app__)

LOCAL_DATA = Path.cwd() / "data"
OCIO_VERSION = "2.0.0beta2"

EXTERNAL_DEPENDENCIES = {
    # "ACES-1.2 Config": dict(
    #     "filename: "OpenColorIO-Config-ACES-1.2.zip",
    #     url="https://github.com/colour-science/OpenColorIO-Configs/"
    #       "releases/download/v1.2/OpenColorIO-Config-ACES-1.2.zip",
    #     size=130123781,
    # ),
    "Marcie ACES2065-1": {
        "filename": "DigitalLAD.2048x1556.exr",
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour/"
        "images/DigitalLAD.2048x1556.exr",
        "size": 25518832,
    },
    "CLF Test Image": {
        "filename": "CLF_testImagePrototype_v006.exr",
        "url": "https://raw.githubusercontent.com/alexfry/CLFTestImage/"
        "master/images/CLF_testImagePrototype_v006.exr",
        "size": 201549,
    },
    "Marcie 4K": {
        "filename": "marcie-4k.exr",
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour/"
        "images/marcie-4k.exr",
        "size": 63015668,
    },
    "OCIO v2.0.0beta2": {
        "filename": "ocio_streamlit_v2.0.0beta2.tar",
        "url": "https://zozobra.s3.us-east-1.amazonaws.com/colour/"
        "data/ocio_streamlit_v2.0.0beta2.tar",
        "size": 28119040,
    },
}
