name = "image_formation_whatever"

version = "0.2.1.0"

description = "Experiments with streamlit, OCIO, colour-science, and more."

authors = ["THE HERMETIC BROTHERHOOD OV SPECTRA"]

requires = [
    "streamlit-0.73.0+<1",
    "OpenColorIO-2.0",
    "colour-0.3.16+",
    "gdown",
    "boltons",
    "plumbum",
    "matplotlib",
    "fs",
    "aenum",
    "pre_commit",
    "rich",
    "vendy",
]

build_command = r"""
mkdir -p $REZ_BUILD_INSTALL_PATH
cp -a {root}/* $REZ_BUILD_INSTALL_PATH
"""


def commands():
    approot = "{root}"
    env.PATH.append("{approot}/bin")
    env.PYTHONPATH.append("{approot}/python")
    setenv("%s_ROOT" % this.name.upper(), "{approot}/")
    setenv("%s_DATA" % this.name.upper(), "{approot}/data")
    alias("ifw", "streamlit run $IMAGE_FORMATION_WHATEVER_ROOT/app.py")
