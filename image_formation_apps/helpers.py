import base64
import os
import sys
import urllib
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from threading import current_thread
from typing import List, Mapping, Optional, Union

import attr
import certifi
import gdown
import streamlit as st
from boltons.fileutils import mkdir_p
from image_formation_toolkit._vendor.munch import Munch
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME

from image_formation_toolkit.settings import EXTERNAL_DEPENDENCIES, LOCAL_DATA, logger


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


@attr.s(auto_attribs=True, frozen=True)
class RemoteData:
    url: str = "https://path/to/resource.exr"
    filename: str = attr.ib()
    size: int = attr.ib()
    family: Optional[str] = None
    label: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Mapping] = None

    @filename.default
    def set_initial_filename(self):
        return self.url.split("/")[-1]

    @size.default
    def set_initial_size(self):
        try:
            with urllib.request.urlopen(self.url, cafile=certifi.where()) as response:
                size = int(response.info()["Content-Length"])
            return size
        except:
            return 0

    def download(self, path=LOCAL_DATA):
        # mostly taken from https://github.com/streamlit/demo-face-gan/
        #   blob/master/streamlit_app.py
        root = Path(path).resolve()
        path = root / self.filename

        # Don't download the file twice. (If possible, verify the
        # download using the file length.)
        if os.path.exists(path):
            if not self.size or os.path.getsize(path) == self.size:
                return path

        mkdir_p(path.parent)

        # These are handles to two visual elements to animate.
        status, progress_bar = None, None
        try:
            status = st.warning("Downloading %s..." % path)

            # handle cases where files hosted on gdrive sometimes fail
            # to download
            if "google.com" in self.url:
                _ = gdown.cached_download(self.url, path=path)
            else:
                progress_bar = st.progress(0)
                # with open(path, "wb") as output_file:
                with urllib.request.urlopen(
                    self.url, cafile=certifi.where()
                ) as response:
                    if response.info()["Content-Length"] is not None:
                        with open(path, "wb") as output_file:
                            length = int(response.info()["Content-Length"])
                            counter = 0.0
                            MEGABYTES = 2.0**20.0
                            while True:
                                data = response.read(8192)
                                if not data:
                                    break
                                counter += len(data)
                                output_file.write(data)

                                # We perform animation by overwriting the elements.
                                status.warning(
                                    "Downloading %s... (%6.2f/%6.2f MB)"
                                    % (path, counter / MEGABYTES, length / MEGABYTES)
                                )
                                progress_bar.progress(min(counter / length, 1.0))

        except urllib.error.URLError as e:
            logger.exception(f"Invalid URL: {self.url}", exc_info=e)
        # Finally, we remove these visual elements by calling .empty().
        finally:
            if status is not None:
                status.empty()
            if progress_bar is not None:
                progress_bar.empty()

        if not path.exists():
            raise FileNotFoundError(str(path))

        elif os.path.getsize(path) == 0:
            os.remove(path)
            raise ValueError(f"Invalid URL: {self.url}")

        return path


def st_file_downloader(bin_file, file_label="File"):
    def get_binary_file_downloader_html(bin_file, file_label="File"):
        # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/27
        with open(bin_file, "rb") as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = (
            f'<a href="data:application/octet-stream;base64,{bin_str}"'
            f' download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        )
        return href

    st.markdown(
        get_binary_file_downloader_html(bin_file, file_label), unsafe_allow_html=True
    )


def get_dependency_local_path(
    key: str, local_dir: Union[str, Path] = LOCAL_DATA
) -> Path:
    remote_file = RemoteData(label=key, **EXTERNAL_DEPENDENCIES[key])
    remote_file.download(path=local_dir)
    return local_dir / remote_file.filename


def get_dependency_data(key: str) -> RemoteData:
    return RemoteData(label=key, **Munch.fromDict(EXTERNAL_DEPENDENCIES[key]))
