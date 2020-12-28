import os
from logging import getLogger
from pathlib import Path
from sysconfig import get_python_version
import sys
import urllib

import attr
from boltons.fileutils import mkdir_p
import fs
from fs.zipfs import ZipFS
import gdown
from plumbum import local
import streamlit as st

logger = getLogger(__name__)


def build_ocio(install_path='/home/appuser', version='2.0.0-beta2',
               build_shared=False, build_apps=False, force=False):
    git = local['git']
    cmake = local['cmake']
    mkdir = local['mkdir']
    rm = local['rm']
    make = local['make']

    def is_ocio_installed():
        # Determine if PyOpenColorIO is already installed.
        python_version = get_python_version()
        pyopencolorio_path = f"{install_path}/lib/python{python_version}/site-packages"
        if local.path(pyopencolorio_path).is_dir():
            if pyopencolorio_path not in sys.path:
                sys.path.append(pyopencolorio_path)
        
        try:
            import PyOpenColorIO
            logger.debug("PyOpenColorIO v{PyOpenColorIO.__version__} is installed.")
            return True
        except ImportError:
            return False


    def archive_ocio_payload(filename='ocio_streamlit.zip'):
        root = fs.open_fs(install_path)
        archive_path = f"{install_path}/{filename}"
        with ZipFS(f"{archive_path}", write=True) as archive:
            fs.copy.copy_dir(root, 'include', archive, install_path)
            fs.copy.copy_dir(root, 'lib', archive, install_path)
            logger.debug(f"Archived {archive_path}")
        return archive_path

    if is_ocio_installed():
        if force:
            rm['-rf'](install_path)
        else:
            return True

    # Configure OCIO build
    releases = ['2.0.0-beta2', '2.0.0-beta1']
    branch = f'v{version}' if version in releases else 'master'
    url = 'https://github.com/AcademySoftwareFoundation/OpenColorIO.git'

    ldflags = f'-Wl,-rpath,{install_path}/lib'
    cxxflags = '-Wno-deprecated-declarations -fPIC'

    cmake_options = [
        #'-G', 'Ninja',
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
    mkdir['-p'](tmp_dir)

    with st.spinner("Building OpenColorIO... patience is a virtue..."):
        with local.cwd(tmp_dir):
            git_repo_path = local.cwd / "OpenColorIO"
            build_dir = local.cwd / 'build'

            # clone release tag (or master branch)
            if not git_repo_path.is_dir():
                logger.debug(f'cloning to {git_repo_path}')
                git['clone', '--branch', branch, url](git_repo_path)

            # clean build dir
            rm['-rf'](build_dir)
            mkdir['-p'](build_dir)

            with local.cwd(build_dir):
                # build and install OCIO
                with local.env(CXXFLAGS=cxxflags, LDFLAGS=ldflags):
                    logger.debug('Invoking CMake...')
                    cmake[cmake_options](git_repo_path)

                    logger.debug('Building and installing...')
                    make['-j1']()
                    make('install')

            _ = archive_ocio_payload()
            logger.info(f"Built and installed OpenColorIO ({branch}): {install_path}")

    if is_ocio_installed():
        # Clean up
        rm['-rf'](tmp_dir)
        return True
    else:
        raise ChildProcessError("Could not install OpenColorIO.")


@attr.s(auto_attribs=True, frozen=True)
class RemoteData:
    filename: str = "my_file.exr"
    url: str = "https://path/to/resource.exr"
    size: int = 0

    def download(self, output_dir=Path.cwd()):
        # mostly taken from https://github.com/streamlit/demo-face-gan/blob/master/streamlit_app.py
        root = Path(output_dir).resolve()
        path = root / self.filename

        # Don't download the file twice. (If possible, verify the download using the file length.)
        if os.path.exists(path):
            if not self.size or os.path.getsize(path) == self.size:
                return path

        mkdir_p(path.parent)

        # These are handles to two visual elements to animate.
        status, progress_bar = None, None
        try:
            status = st.warning("Downloading %s..." % path)

            # handle cases where files hosted on gdrive someitimes fail to download
            if "google.com" in self.url:
                _ = gdown.cached_download(self.url, path=path)
            else:
                progress_bar = st.progress(0)
                with open(path, "wb") as output_file:
                    with urllib.request.urlopen(self.url) as response:
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
                            status.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                                           (path, counter / MEGABYTES, length / MEGABYTES))
                            progress_bar.progress(min(counter / length, 1.0))

        # Finally, we remove these visual elements by calling .empty().
        finally:
            if status is not None:
                status.empty()
            if progress_bar is not None:
                progress_bar.empty()

        return path
