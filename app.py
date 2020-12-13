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

__author__ = "dcac@deadlythings.com"
__license__ = "GPL-3"
__version__ = "0.1.0"

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


demo_pages = {
    "Test": test,
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

