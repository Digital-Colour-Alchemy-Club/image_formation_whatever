from contextlib import contextmanager
from io import StringIO
from logging import getLogger
import sys
from threading import current_thread

import streamlit as st
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME

logger = getLogger(__name__)


@contextmanager
def st_redirect(src, dst):
    """
    Redirects an output stream to a target streamlit write function.
    """
    # https://discuss.streamlit.io/t/cannot-print-the-terminal-output-in-streamlit/6602/9
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield


