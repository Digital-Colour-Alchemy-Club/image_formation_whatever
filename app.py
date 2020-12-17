import sys
from pathlib import Path
import streamlit as st
import numpy as np
import colour


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass

_LIBRARY_VERSIONS = {}

try:
    import OpenImageIO as oiio
    _LIBRARY_VERSIONS['OpenImageIO'] = oiio.__version__
except ImportError:
    _LIBRARY_VERSIONS['OpenImageIO'] = None

try:
    import PyOpenColorIO as ocio
    _LIBRARY_VERSIONS['OpenColorIO'] = ocio.__version__
except ImportError:
    _LIBRARY_VERSIONS['OpenColorIO'] = None
    

__author__ = "dcac@deadlythings.com"
__license__ = "GPL3"
__version__ = "0.1.1"

VERSION = '.'.join(__version__.split('.')[:2])


st.title("Image Formation Whatever")


intro = """
# Markdown
## Stuff
### Here

And here.

And...
 - Here?
 - Here.
 - Here!

But also with emoji stuff like :rainbow: and 👋.
"""

def draw_main_page():
    st.write(f"""
    # Here's a thing that does stuff! 👋
    """)

    st.write(intro)

    st.info("""
        :point_left: **To get started, choose a demo on the left sidebar.**
    """)
    st.balloons()


def test():
    st.write("""
    # Testes, testes...

    1.. 2...

    ...3?

    """)


def about():
    st.write(colour.utilities.describe_environment())
    for lib in ["OpenColorIO", "OpenImageIO"]:
        if lib in _LIBRARY_VERSIONS.keys():
            st.write(lib): _LIBRARY_VERSIONS[lib]


demo_pages = {
    "Teste": test,
    'About': about,
}

# Draw sidebar
pages = list(demo_pages.keys())
pages.insert(0, "Release Notes")

st.sidebar.title(f"Demos v{VERSION}")
selected_demo = st.sidebar.radio("", pages)

# Draw main page
if selected_demo in demo_pages:
    demo_pages[selected_demo]()
else:
    draw_main_page()






# External files to download.
EXTERNAL_DEPENDENCIES = {
    "OpenColorIO-Config-ACES-1.2.zip": {
        "url": "https://github.com/colour-science/OpenColorIO-Configs/releases/download/v1.2/OpenColorIO-Config-ACES-1.2.zip",
        "size": 130123781
    },
    "DigitalLAD.2048x1556.exr": {
        "url": "https://drive.google.com/uc?id=1GrltrT4cb8PPhVIMII4fWRgIgAPdrpi",
        "size": 2551883
    },
    "CLF_testImagePrototype_v006.exr": {
        "url": "https://github.com/alexfry/CLFTestImage/blob/master/images/CLF_testImagePrototype_v006.exr",
        "size": 201549
    },
}
