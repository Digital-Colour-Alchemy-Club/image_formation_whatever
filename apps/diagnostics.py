import sys

import colour
import fs
import streamlit as st
from boltons.ecoutils import get_profile

from helpers import st_stdout


def diagnostics(local_data="."):
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
    st.write(local_data)
    with st_stdout("code"):
        fs.open_fs(str(local_data)).tree()
