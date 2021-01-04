import sys

import colour
import fs
import streamlit as st
from boltons.ecoutils import get_profile

from app import LOCAL_DATA
from util import st_stdout


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