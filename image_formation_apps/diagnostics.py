import sys

import colour
import fs
import streamlit as st
from boltons.ecoutils import get_profile

from image_formation_apps.helpers import st_stdout


def diagnostics(local_data="."):
    st.header("Streamlit instance info")
    st.subheader("Locally-installed libraries")
    try:
        import PyOpenColorIO as ocio

        st.write(f"PyOpenColorIO v{ocio.__version__}")
    except ImportError:
        pass
    try:
        import OpenImageIO as oiio

        st.write(f"OpenImageIO v{oiio.__version__}")
    except:
        pass

    st.subheader("`colour-science` library info")
    with st_stdout("code"):
        colour.utilities.describe_environment()

    st.subheader("System")
    st.write(get_profile())

    st.subheader("Local contents")
    with st_stdout("code"):
        print(str(local_data))
        fs.open_fs(str(local_data)).tree()

    st.subheader("Python system paths")
    st.write(sys.path)
