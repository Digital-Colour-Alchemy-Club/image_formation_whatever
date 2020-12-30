from contextlib import contextmanager
from io import StringIO
from logging import getLogger
import os
from pathlib import Path
import sys
from sysconfig import get_python_version
from threading import current_thread
import urllib

import attr
from boltons.fileutils import mkdir_p
import certifi
import gdown
from plumbum import local
import streamlit as st
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME

logger = getLogger(__name__)

VALID_OCIO_VERSIONS = ['2.0.0-beta2', '2.0.0-beta1']
LOCAL_INSTALL_PREFIX = os.path.expanduser("~/")


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


def build_ocio(install_path=LOCAL_INSTALL_PREFIX,
               version='2.0.0-beta2',
               build_shared=False,
               build_apps=False,
               force=False):
    """
    Builds and installs OpenColorIO.

    Parameters
    ----------
    install_path : unicode, optional
        Destination directory for installed libraries and headers
    version : unicode, optional
        Library version to build and install. If value does not match a known
        version, the main branch of the github repository will be used.
    build_shared : bool, optional
        Whether to build shared libraries instead of static.
    build_apps : bool, optional
        Whether to build and install cli applications.
    force : bool, optional
        Whether to force a re-build and re-install, even if the library is
        already installed.

    Returns
    -------
    bool
        Definition success.

    """

    def is_ocio_installed():
        """
        Checks if `PyOpenColorIO` is installed and importable.
        """
        python_version = get_python_version()
        pyopencolorio_path = \
            f"{install_path}/lib/python{python_version}/site-packages"
        if local.path(pyopencolorio_path).is_dir():
            if pyopencolorio_path not in sys.path:
                sys.path.append(pyopencolorio_path)
        try:
            import PyOpenColorIO
            logger.debug(
                "PyOpenColorIO v{PyOpenColorIO.__version__} is installed.")
            return True
        except ImportError:
            return False

    def archive_ocio_payload(filename='ocio_streamlit.tar'):
        """
        Creates a compressed archive of the compiled library and headers.
        """
        archive_path = f"{install_path}/{filename}"
        local['tar']['-cf',
                     f"{archive_path}",
                     f"{install_path}/lib",
                     f"{install_path}/include"]()
        logger.debug(f"Archived {archive_path}")
        return archive_path

    if is_ocio_installed():
        if force:
            # Clean prefix and proceed with build + install
            python_version = get_python_version()
            logger.debug("Removing existing OCIO artifacts...")
            paths = [f"{install_path}/lib/python{python_version}"
                     f"/site-packages/PyOpenColorIO.so",
                     f"{install_path}/lib/libOpenColorIO*",
                     f"{install_path}/include/OpenColorIO",
                     f"{install_path}/bin/ocio*"]
            _ = [local['rm']['-rf'](p) for p in paths]
        else:
            # Bypass build + install.
            return True

    # Configure build variables
    branch = f'v{version}' if version in VALID_OCIO_VERSIONS else 'master'
    url = 'https://github.com/AcademySoftwareFoundation/OpenColorIO.git'

    ldflags = f'-Wl,-rpath,{install_path}/lib'
    cxxflags = '-Wno-deprecated-declarations -fPIC'

    cmake_options = [
        # '-G', 'Ninja',
        f'-DOCIO_BUILD_APPS={build_apps}',
        '-DOCIO_BUILD_NUKE=OFF',
        '-DOCIO_BUILD_DOCS=OFF',
        '-DOCIO_BUILD_TESTS=OFF',
        '-DOCIO_BUILD_GPU_TESTS=OFF',
        '-DOCIO_USE_HEADLESS=ON',
        '-DOCIO_BUILD_PYTHON=ON',
        '-DOCIO_BUILD_JAVA=OFF',
        f'-DBUILD_SHARED_LIBS={build_shared}',
        f'-DCMAKE_INSTALL_PREFIX={install_path}',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CXX_STANDARD=14',
        '-DOCIO_INSTALL_EXT_PACKAGES=MISSING',
        '-DOCIO_WARNING_AS_ERROR=OFF',
    ]

    # create temporary dir for building
    tmp_dir = local.path('/tmp', 'build')
    local['mkdir']['-p'](tmp_dir)

    with st.spinner("Building OpenColorIO... patience is a virtue..."):
        with local.cwd(tmp_dir):
            git_repo_path = local.cwd / "OpenColorIO"
            build_dir = local.cwd / 'build'

            # clone release tag (or master branch)
            if not git_repo_path.is_dir():
                logger.debug(f'cloning to {git_repo_path}')
                local['git']['clone', '--branch', branch, url](git_repo_path)

            # clean build dir
            local['rm']['-rf'](build_dir)
            local['mkdir']['-p'](build_dir)

            with local.cwd(build_dir):
                # build and install OCIO
                with local.env(CXXFLAGS=cxxflags, LDFLAGS=ldflags):
                    logger.debug('Invoking CMake...')
                    local['cmake'][cmake_options](git_repo_path)

                    logger.debug('Building and installing...')
                    local['make']['-j1']()
                    local['make']('install')

            _ = archive_ocio_payload()
            logger.info(
                f"Built and installed OpenColorIO ({branch}): {install_path}")

    if is_ocio_installed():
        # Clean up build dir
        local['rm']['-rf'](tmp_dir)
        return True
    else:
        raise ChildProcessError("Could not install OpenColorIO.")


@attr.s(auto_attribs=True, frozen=True)
class RemoteData:
    filename: str = "my_file.exr"
    url: str = "https://path/to/resource.exr"
    size: int = 0

    def download(self, output_dir=Path.cwd()):
        # mostly taken from https://github.com/streamlit/demo-face-gan/
        #   blob/master/streamlit_app.py
        root = Path(output_dir).resolve()
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
                with open(path, "wb") as output_file:
                    with urllib.request.urlopen(
                            self.url, cafile=certifi.where()) as response:
                        length = int(response.info()["Content-Length"])
                        counter = 0.0
                        MEGABYTES = 2.0 ** 20.0
                        while True:
                            data = response.read(8192)
                            if not data:
                                break
                            counter += len(data)
                            output_file.write(data)

                            # We perform animation by overwriting the elements.
                            status.warning(
                                "Downloading %s... (%6.2f/%6.2f MB)" %
                                (
                                    path,
                                    counter / MEGABYTES,
                                    length / MEGABYTES
                                )
                            )
                            progress_bar.progress(min(counter / length, 1.0))

        # Finally, we remove these visual elements by calling .empty().
        finally:
            if status is not None:
                status.empty()
            if progress_bar is not None:
                progress_bar.empty()

        return path
