import re
import sys
from pathlib import Path
from sysconfig import get_python_version

import fs
import streamlit as st
from plumbum import local

from helpers import (
    get_dependency_local_path,
    get_dependency_data,
    logger,
    st_file_downloader,
    st_stdout,
)
from settings import config


def setup_opencolorio(
    prefix=config.libs.prefix, version=config.libs.OpenColorIO.version, force=False
):
    try:
        import PyOpenColorIO as ocio
    except ImportError:

        class Null(object):
            def __getattr__(self, name):
                return None

            def __bool__(self):
                return False

        ocio = Null()

    def install_opencolorio(prefix=prefix, version=version, force=force):
        with st.spinner("Setting up OpenColorIO..."):
            if force:
                build_ocio(
                    prefix=prefix,
                    version=version,
                    force=force,
                    build_apps=True,
                    build_shared=False,
                )
            else:
                try:
                    fetch_ocio(version=version, force=force)
                except Exception as e:
                    raise e
                    build_ocio(
                        prefix=prefix,
                        version=version,
                        force=force,
                        build_apps=True,
                        build_shared=False,
                    )

    # Offer archive of existing libraries
    if ocio:
        lib_archive = Path(ocio.__file__).parents[3] / "ocio_streamlit.tar"

        if lib_archive.exists():
            # archive generated at build time (see `build_ocio` method)
            st_file_downloader(lib_archive, f"OCIO v{ocio.__version__} libs")

    install_opencolorio(prefix=prefix, version=version, force=False)


def update_imageio():
    # Install imageio freeimage plugin (i.e., for EXR support)
    import imageio

    imageio.plugins.freeimage.download()


def run_bootstrap():
    update_imageio()
    setup_opencolorio()


def build_ocio(
    prefix=config.libs.prefix,
    version=config.libs.OpenColorIO.version,
    build_shared=False,
    build_apps=False,
    force=False,
):
    """
    Builds and installs OpenColorIO.

    Parameters
    ----------
    prefix : unicode, optional
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
        pyopencolorio_path = f"{prefix}/lib/python{python_version}/site-packages"
        if local.path(pyopencolorio_path).is_dir():
            if pyopencolorio_path not in sys.path:
                sys.path.append(pyopencolorio_path)
        try:
            import PyOpenColorIO

            logger.debug(f"PyOpenColorIO v{PyOpenColorIO.__version__} is installed.")
            return True
        except ImportError:
            return False

    def archive_ocio_payload(output_dir=prefix, filename="ocio_streamlit.tar"):
        """
        Creates a compressed archive of the compiled library and headers.
        """
        archive_path = f"{output_dir}/{filename}"
        src = fs.open_fs(f"{prefix}")
        glob_filters = ["**/OpenColorIO/**/*.h", "**/ocio*", "**/lib/**/*penColor*"]
        files_to_archive = [p.path for g in glob_filters for p in src.glob(g)]
        with fs.open_fs(f"tar:///{archive_path}", writeable=True, create=True) as dst:
            for file in files_to_archive:
                fs.copy.copy_file(src, file, dst, file)
        logger.debug(f"Archived {archive_path}")
        return archive_path

    if is_ocio_installed():
        if force:
            # Clean prefix and proceed with build + install
            logger.debug("Removing existing OCIO artifacts...")
            install_root = fs.open_fs(f"{prefix}")
            glob_filters = ["**/OpenColorIO/**/*.h", "**/ocio*", "**/lib/**/*penColor*"]
            _ = [p.remove() for g in glob_filters for p in install_root.glob(g)]

        else:
            # Bypass build + install
            return True

    mkdir, cmake, make, rm, git, tar = [
        local[bin] for bin in ["mkdir", "cmake", "make", "rm", "git", "tar"]
    ]

    # Configure build variables
    url = "https://github.com/AcademySoftwareFoundation/OpenColorIO.git"
    git_clone = git["clone"]
    if version in ["2.0.0beta1", "2.0.0beta2", "2.0.0rc1"]:
        branch = re.sub(r"(\d)(\w)", r"\1-\2", f"v{version}")
        git_clone = git_clone["--branch", branch]
    git_clone = git_clone[url]

    ldflags = f"-Wl,-rpath,{prefix}/lib"
    cxxflags = "-Wno-deprecated-declarations -fPIC"

    cmake_options = [
        # '-G', 'Ninja',
        f"-DOCIO_BUILD_APPS={build_apps}",
        "-DOCIO_BUILD_NUKE=OFF",
        "-DOCIO_BUILD_DOCS=OFF",
        "-DOCIO_BUILD_TESTS=OFF",
        "-DOCIO_BUILD_GPU_TESTS=OFF",
        "-DOCIO_USE_HEADLESS=ON",
        "-DOCIO_BUILD_PYTHON=ON",
        "-DOCIO_BUILD_JAVA=OFF",
        f"-DBUILD_SHARED_LIBS={build_shared}",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=14",
        "-DOCIO_INSTALL_EXT_PACKAGES=MISSING",
        "-DOCIO_WARNING_AS_ERROR=OFF",
    ]

    # create temporary dir for building
    tmp_dir = "/tmp/build"
    mkdir["-p"](tmp_dir)

    with local.cwd(tmp_dir):
        git_repo_path = local.cwd / "OpenColorIO"
        build_dir = local.cwd / "build"

        # clone release tag (or master branch)
        if not git_repo_path.is_dir():
            logger.debug(f"cloning to {git_repo_path}")
            git_clone(git_repo_path)

        # clean build dir
        rm["-rf"](build_dir)
        mkdir["-p"](build_dir)

        with local.cwd(build_dir):
            # build and install OCIO
            with local.env(CXXFLAGS=cxxflags, LDFLAGS=ldflags):
                logger.debug("Invoking CMake...")
                cmake[cmake_options](git_repo_path)

                logger.debug("Building and installing...")
                make["-j1"]("install")

        _ = archive_ocio_payload()
        logger.info(f"Built and installed OpenColorIO ({version}): {prefix}")

    if is_ocio_installed():
        # Clean up build dir
        fs.open_fs(tmp_dir).removetree(".")
        return True
    else:
        raise ChildProcessError("Could not install OpenColorIO.")


def fetch_ocio(version=config.libs.OpenColorIO.version, force=False):

    # Only download if not importable
    if not force:
        try:
            import PyOpenColorIO

            if version == PyOpenColorIO.__version__:
                return
        except ImportError:
            pass

    # fetch archive
    key = f"OCIO v{version}"
    archive_path = get_dependency_local_path(key)
    archive = fs.open_fs(f"tar://{archive_path}")
    prefix = get_dependency_data(key).metadata.prefix
    # unpack
    with fs.open_fs(prefix, writeable=True) as dst:
        fs.copy.copy_dir(archive, prefix, dst, ".")

    # append to system path
    dst = fs.open_fs(prefix)
    sys.path.extend([dst.getsyspath(p.path) for p in dst.glob("**/site-packages/")])

    # validate
    try:
        import PyOpenColorIO
    except ImportError:
        with st_stdout("error"):
            print(
                """***NICE WORK, GENIUS: ***
            You've managed to build, archive, retrieve, and deploy OCIO...
            yet you couldn't manage to import PyOpenColorIO."""
            )

    logger.debug(f"OpenColorIO v{version} installed!")
