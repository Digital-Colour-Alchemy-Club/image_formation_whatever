import logging
import re
import sys
from sysconfig import get_python_version

import colour
import fs
import PyOpenColorIO as ocio
from plumbum import local
import numpy as np
from six import string_types


from util import logger, st_stdout
from data_utilities import get_dependency


def build_ocio(prefix="/usr/local",
               version='2.0.0beta2',
               build_shared=False,
               build_apps=False,
               force=False):
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
        pyopencolorio_path = \
            f"{prefix}/lib/python{python_version}/site-packages"
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

    def archive_ocio_payload(output_dir=prefix, filename='ocio_streamlit.tar'):
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
        local[bin] for bin in [
            'mkdir', 'cmake', 'make', 'rm', 'git', 'tar']
    ]

    # Configure build variables
    url = 'https://github.com/AcademySoftwareFoundation/OpenColorIO.git'
    git_clone = git['clone']
    if version in ['2.0.0beta1', '2.0.0beta2']:
        branch = re.sub(r"(\d)(\w)", r"\1-\2", f'v{version}')
        git_clone = git_clone['--branch', branch]
    git_clone = git_clone[url]

    ldflags = f'-Wl,-rpath,{prefix}/lib'
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
        f'-DCMAKE_INSTALL_PREFIX={prefix}',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CXX_STANDARD=14',
        '-DOCIO_INSTALL_EXT_PACKAGES=MISSING',
        '-DOCIO_WARNING_AS_ERROR=OFF',
    ]

    # create temporary dir for building
    tmp_dir = '/tmp/build'
    mkdir['-p'](tmp_dir)

    with local.cwd(tmp_dir):
        git_repo_path = local.cwd / "OpenColorIO"
        build_dir = local.cwd / 'build'

        # clone release tag (or master branch)
        if not git_repo_path.is_dir():
            logger.debug(f'cloning to {git_repo_path}')
            git_clone(git_repo_path)

        # clean build dir
        rm['-rf'](build_dir)
        mkdir['-p'](build_dir)

        with local.cwd(build_dir):
            # build and install OCIO
            with local.env(CXXFLAGS=cxxflags, LDFLAGS=ldflags):
                logger.debug('Invoking CMake...')
                cmake[cmake_options](git_repo_path)

                logger.debug('Building and installing...')
                make['-j1']('install')

        _ = archive_ocio_payload()
        logger.info(
            f"Built and installed OpenColorIO ({version}): {prefix}")

    if is_ocio_installed():
        # Clean up build dir
        fs.open_fs(tmp_dir).removetree('.')
        return True
    else:
        raise ChildProcessError("Could not install OpenColorIO.")


def fetch_ocio(prefix="/home/appuser",
               version="2.0.0beta2",
               force=False):

    # Only download if not importable
    if not force:
        try:
            import PyOpenColorIO
            if version == PyOpenColorIO.__version__:
                return
        except ImportError:
            pass

    # fetch archive
    archive_path = get_dependency(f"OCIO v{version}")
    archive = fs.open_fs(f"tar://{archive_path}")

    # unpack
    with fs.open_fs(prefix, writeable=True) as dst:
        fs.copy.copy_dir(archive, prefix, dst, '.')

    # append to system path
    dst = fs.open_fs(prefix)
    sys.path.extend([
        dst.getsyspath(p.path) for p in dst.glob("**/site-packages/")
    ])

    # validate
    try:
        import PyOpenColorIO
    except ImportError:
        with st_stdout("error"):
            print("""***NICE WORK, GENIUS: ***
            You've managed to build, archive, retrieve, and deploy OCIO...
            yet you couldn't manage to import PyOpenColorIO.""")

    logger.debug(f"OpenColorIO v{version} installed!")


import PyOpenColorIO as OCIO
from aenum import Enum, MultiValueEnum

class Displays(Enum):
    REC_709 = "Rec.709"
    REC_2020 = "Rec.2020"
    SRGB = "sRGB"
    P3_DCI = "P3-DCI"
    P3_D60 = "P3-D60"
    P3_D65 = "P3-D65"
    DCDM = "DCDM"
    REC_2100_PQ = "Rec.2100-PQ"
    REC_2100_HLG = "Rec.2100-HLG"
    ST_2084_P3_D65 = "ST-2084 P3-D65"


class DisplayCCTFs(Enum):
    L_STAR = "L*"
    SRGB = "sRGB"
    REC709 = "Rec709 (Display)"
    GAMMA26 = "Gamma 2.6"
    GAMMA24 = "Gamma 2.4"
    GAMMA22 = "Gamma 2.2"
    GAMMA18 = "Gamma 1.8"
    PFWL = 'ARRI Gamma ~2.4'
    PFWL26 = 'ARRI Gamma ~2.6'
    ADOBE_1998 = "Adobe (1998)"

    @property
    def decoding_xform(self):
        cls = DisplayCCTFs
        return {
            cls.L_STAR: display_cctf(gamma=3., offset=0.16),
            cls.SRGB: display_cctf(gamma=2.4, offset=0.055),
            cls.REC709: display_cctf(gamma=1/0.45, offset=0.099),
            cls.GAMMA26: display_cctf(gamma=2.6),
            cls.GAMMA24: display_cctf(gamma=2.4),
            cls.GAMMA22: display_cctf(gamma=2.2),
            cls.GAMMA18: display_cctf(gamma=1.8),
            cls.PFWL: display_cctf(gamma=2.725, offset=0.097),
            cls.PFWL26: display_cctf(gamma=2.9961, offset=0.1111),
            cls.ADOBE_1998: display_cctf(gamma=563/256.),
        }[self]

    @property
    def encoding_xform(self):
        cls = DisplayCCTFs
        xf = self.decoding_xform
        xf.setDirection(ocio.TRANSFORM_DIR_INVERSE)
        return xf




DISPLAY_BUILTINS = {
    Displays.REC_709: 'CIE-XYZ-D65_to_Display-REC709',
    Displays.REC_2020: 'CIE-XYZ-D65_to_Display-REC2020',
    Displays.SRGB: 'CIE-XYZ-D65_to_Display-sRGB',
    Displays.P3_DCI: 'CIE-XYZ-D65_to_Display-P3DCI-BFD',
    Displays.P3_D60: 'CIE-XYZ-D65_to_Display-P3D60-BFD',
    Displays.P3_D65: 'CIE-XYZ-D65_to_Display-P3D65',
    Displays.DCDM: 'CIE-XYZ-D65_to_Display-DCDM',
    Displays.REC_2100_PQ: 'CIE-XYZ-D65_to_Display-REC2100-PQ',
    Displays.REC_2100_HLG: 'CIE-XYZ-D65_to_Display-REC2100-HLG',
    Displays.ST_2084_P3_D65: 'CIE-XYZ-D65_to_Display-ST2084-P3D65',
}


def aces_viewtransform_factory(display=Displays.SRGB, d60sim=False, legal=False, limit=None, ):
    return NotImplemented


# helpers

def collapse_redundant_channels(table, max_channels=3):
    t = np.copy(table)
    t *= np.ones(max_channels)
    if np.allclose(*[t[..., c] for c in range(max_channels)]):
        return t[..., 0]
    return t

def _rgba_tuple(x, alpha=1.):
    return [x] * 3 + [float(alpha)]


def get_name(obj):
    """Retrieve the name of an OCIO object"""
    if isinstance(obj, string_types):
        return obj

    return obj.getName()

# transform helpers
def display_cctf(gamma=1., offset=None, direction=None, negative_style=None):
    if offset:
        cctf = ocio.ExponentWithLinearTransform()
        cctf.setGamma(_rgba_tuple(gamma))
        cctf.setOffset(_rgba_tuple(offset, alpha=0.))
    else:
        cctf = ocio.ExponentTransform(_rgba_tuple(gamma))

    if direction:
        cctf.setDirection(direction)
    if negative_style:
        cctf.setNegativeStyle(negative_style)

    return cctf






def _simplify_transform(transform, optimization=ocio.OPTIMIZATION_VERY_GOOD,
                        direction=ocio.TRANSFORM_DIR_FORWARD,
                        config=None, context=None):
    config = config or ocio.GetCurrentConfig()
    context = context or config.getCurrentContext()
    proc = config.getProcessor(transform=transform, context=context, direction=direction)
    gt = proc.getOptimizedProcessor(optimization).createGroupTransform()
    return gt[0] if len(gt) < 2 else gt


def _convert_numpy_matrix33_to_matrix_transform(matrix33, direction=ocio.TRANSFORM_DIR_FORWARD):
    mtx = np.pad(np.array(matrix33).flatten().reshape(3, 3), [(0, 1), (0, 1)]).flatten()
    mtx[-1] = 1
    return ocio.MatrixTransform(matrix=mtx, direction=direction)


def _convert_matrix_transform_to_numpy_matrix33(matrix_transform, direction=ocio.TRANSFORM_DIR_FORWARD):
    mt = (_simplify_transform(matrix_transform, direction=direction))
    return np.array(mt.getMatrix()).reshape(4, 4)[:3, :3]




def get_camera_colorspaces(family=''):
    colorspaces = []
    # get Camera Log colorspaces from Builtins
    for name, description in ocio.BuiltinTransformRegistry().getBuiltins():
        if name.endswith('to_ACES2065-1'):
            cs = ocio.ColorSpace(
                name=description.split('Convert ')[-1].split(' to ')[0],
                family=family,
                encoding='log',
                toReference=ocio.BuiltinTransform(name))
            colorspaces.append(cs)
    return colorspaces




def get_display_colorspaces(family=''):
    colour.RGB_COLOURSPACES['CIE-XYZ (D65)'] = colour.RGB_Colourspace(
        name='CIE-XYZ (D65)',
        primaries=np.array([[1., 0.], [0., 1.], [0., 0.]]),
        whitepoint=np.array([0.3127, 0.329]),
        whitepoint_name='D65',
        use_derived_matrix_RGB_to_XYZ=True,
        use_derived_matrix_XYZ_to_RGB=True,
        cctf_decoding=colour.models.linear_function,
        cctf_encoding=colour.models.linear_function,
    )

    xyzd65_to_rec709 = _simplify_transform(ocio.GroupTransform(
       [ocio.BuiltinTransform('ACES-AP1_to_CIE-XYZ-D65_BFD', ocio.TRANSFORM_DIR_INVERSE),
        ocio.BuiltinTransform('ACES-AP1_to_LINEAR-REC709_BFD')]))

    # xyzd65_to_rec709 = _convert_numpy_matrix33_to_matrix_transform(
    #     colour.matrix_RGB_to_RGB(
    #         input_colourspace=colour.RGB_COLOURSPACES['CIE-XYZ (D65)'],
    #         output_colourspace=colour.RGB_COLOURSPACES['ITU-R BT.709'],
    #         chromatic_adaptation_transform=None
    #     )
    # )

    xyzd65_to_p3d65 = _convert_numpy_matrix33_to_matrix_transform(
        colour.matrix_RGB_to_RGB(
            input_colourspace=colour.RGB_COLOURSPACES['CIE-XYZ (D65)'],
            output_colourspace=colour.RGB_COLOURSPACES['P3-D65'],
            chromatic_adaptation_transform=None
        )
    )

    xyzd65_to_p3 = _convert_numpy_matrix33_to_matrix_transform(
        colour.matrix_RGB_to_RGB(
            input_colourspace=colour.RGB_COLOURSPACES['CIE-XYZ (D65)'],
            output_colourspace=colour.RGB_COLOURSPACES['DCI-P3'],
            chromatic_adaptation_transform='Bradford',
        )
    )

    # check
    xyzd65_to_p3_d60_sim_bfd = ocio.GroupTransform(
        [
            ocio.BuiltinTransform('ACES-AP0_to_CIE-XYZ-D65_BFD', ocio.TRANSFORM_DIR_INVERSE),
            _convert_numpy_matrix33_to_matrix_transform(
                colour.matrix_RGB_to_RGB(
                    input_colourspace=colour.RGB_COLOURSPACES['ACES2065-1'],
                    output_colourspace=colour.RGB_COLOURSPACES['DCI-P3'],
                    chromatic_adaptation_transform='Bradford',
                )
            )
        ]
    )

    # Shapers
    acescct_cctf = ocio.BuiltinTransform('ACEScct-LOG_to_LIN')


    sRGB_g22 = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        name="sRGB (Gamma ~2.2)",
        description="sRGB (Gamma ~2.2) with Rec.709 primaries",
        encoding='sdr-video',
        family=family,
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform([
            xyzd65_to_rec709,
            display_cctf(gamma=2.4, offset=0.055, direction=ocio.TRANSFORM_DIR_INVERSE)
        ]),
    )

    sRGB = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        name="sRGB",
        description="sRGB with Rec.709 primaries",
        encoding='sdr-video',
        family=family,
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform([
            xyzd65_to_rec709,
            display_cctf(gamma=2.2, direction=ocio.TRANSFORM_DIR_INVERSE)
        ]),
    )
    Rec709_display = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        name="Rec.709 (legacy)",
        description="ITU-R BT.709 inverse camera CCTF with BT.709 primaries",
        encoding='sdr-video',
        family=family,
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform([
            xyzd65_to_rec709,
            display_cctf(gamma=1/0.45, offset=0.099, direction=ocio.TRANSFORM_DIR_INVERSE)
        ]),
    )

    Rec1886 = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        name="Rec.1886",
        description="ITU-R BT.1886 with BT.709 primaries and D65 whitepoint",
        encoding='sdr-video',
        family=family,
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform([
            xyzd65_to_rec709,
            display_cctf(gamma=2.4, direction=ocio.TRANSFORM_DIR_INVERSE)
        ]),
    )

    DCI_P3 = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        name="DCI-P3",
        description="Gamma 2.6 with DCI primaries",
        encoding='sdr-video',
        family=family,
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform([
            xyzd65_to_p3,
            display_cctf(gamma=2.6, direction=ocio.TRANSFORM_DIR_INVERSE)
        ]),
    )

    P3_D65 = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        name="P3-D65",
        description="Gamma 2.6 with DCI primaries and D65 whitepoint",
        encoding='sdr-video',
        family=family,
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform([
            xyzd65_to_p3d65,
            display_cctf(gamma=2.6, direction=ocio.TRANSFORM_DIR_INVERSE)
        ]),
    )
    # TODO: D60 sim; hdr colorspaces

    return [
        sRGB,
        Rec1886,
        DCI_P3,
        P3_D65,
    ]






def _add_default_colorspaces_and_roles(config=None, scene_linear='ACEScg'):
    # create default colorspaces

    linear_allocation_vars = [-8, 5, 0.00390625]
    # data / no-color
    raw = ocio.ColorSpace(name='raw', isData=True, equalityGroup='nc', encoding='data')

    # Scene and display reference
    aces = ocio.ColorSpace(
        name='ACES2065-1',
        equalityGroup='ap0',
        encoding='scene-linear',
        allocation=ocio.ALLOCATION_LG2,
        allocationVars=linear_allocation_vars,
    )

    xyzd65 = ocio.ColorSpace(
        name='CIE-XYZ (D65)',
        equalityGroup='xyz',
        encoding='display-linear',
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        allocation=ocio.ALLOCATION_LG2,
        allocationVars=linear_allocation_vars,
    )

    # Other linear reference spaces
    acescg = ocio.ColorSpace(
        name='ACEScg',
        equalityGroup='ap1',
        toReference=ocio.BuiltinTransform('ACEScg_to_ACES2065-1'),
        encoding='scene-linear',
        allocation=ocio.ALLOCATION_LG2,
        allocationVars=linear_allocation_vars,
    )

    lin709 = ocio.ColorSpace(
        name='Linear-Rec709',
        encoding='scene-linear',
        toReference=ocio.GroupTransform(
            [ocio.BuiltinTransform('ACES-AP1_to_LINEAR-REC709_BFD', direction=ocio.TRANSFORM_DIR_INVERSE),
             ocio.BuiltinTransform('ACEScg_to_ACES2065-1')]),
        allocation=ocio.ALLOCATION_LG2,
        allocationVars=linear_allocation_vars,
    )

    cfg = config or ocio.GetCurrentConfig()

    # add default colorspaces to config and update roles
    _ = [cfg.addColorSpace(cs) for cs in [raw, aces, xyzd65, acescg, lin709]]

    scene_linear = {
        acescg.getName(): acescg,
        lin709.getName(): lin709
    }[scene_linear]

    cfg.setRole('aces_interchange', aces.getName())
    cfg.setRole('cie_xyz_d65_interchange', xyzd65.getName())
    cfg.setRole(ocio.ROLE_DATA, raw.getName())
    cfg.setRole(ocio.ROLE_DEFAULT, scene_linear.getName())
    cfg.setRole(ocio.ROLE_REFERENCE, aces.getName())
    cfg.setRole(ocio.ROLE_SCENE_LINEAR, scene_linear.getName())

    # add default colorimetry view transform to config
    cfg.addViewTransform(
        ocio.ViewTransform(
            name='colorimetry',
            referenceSpace=ocio.REFERENCE_SPACE_SCENE,
            fromReference=ocio.BuiltinTransform('ACES-AP0_to_CIE-XYZ-D65_BFD'))
        )
    return cfg






def create_new_config(scene_linear='ACEScg', config=None):
    cfg = _add_default_colorspaces_and_roles(scene_linear=scene_linear, config=config)

    default_spaces = list(cfg.getColorSpaces())
    cam_log_spaces = get_camera_colorspaces()
    display_spaces = get_display_colorspaces()

    _ = [cfg.addColorSpace(cs) for cs in cam_log_spaces + display_spaces]

    # file rules
    filerules = ocio.FileRules()
    filerules.setDefaultRuleColorSpace(scene_linear)
    filerules.insertPathSearchRule(0)
    cfg.setFileRules(filerules)

    # todo: add default displays

    default_display = Displays.SRGB
    #cfg.addDisplayView(default_display.value, 'Colorimetry','Colorimetry')
    cfg.addSharedView('Colorimetry', 'colorimetry', '<USE_DISPLAY_NAME>')
    cfg.addDisplaySharedView(default_display.value, 'Colorimetry')
    return cfg


def serialize_processor(processor, format='Color Transform Format', filename=None):
    if ('clf' in str(format).lower() or
        str(format).lower().startswith('academy') or
        str(filename).lower().endswith('clf')):
        format = 'Academy/ASC Common LUT Format'
    if filename:
        #TODO: return CLF stuff
        processor.write(format, filename)
    return processor.write(format)



def ocio_colourspace_process(a, config, source, target=None, display=None, view=None, use_cpu=True):
    # modified version of TM's code, adds "display-view" processes
    a = colour.utilities.as_float_array(a, dtype=np.float32)

    height, width, channels = a.shape

    if target:
        processor = config.getProcessor(source, target)
    elif display and view:
        processor = config.getProcessor(source, display, view)
    else:
        raise AttributeError

    processor = (
        processor.getDefaultCPUProcessor()
        if use_cpu else
        processor.getDefaultGPUProcessor()
    )

    image_desc = ocio.PackedImageDesc(a,
        width, height, channels)

    processor.apply(image_desc)

    return image_desc.getData().reshape([height, width, channels])



def baby_config():

    linear = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_SCENE,
        name="linear",
        description="scene-linear",
        encoding='scene-linear',
        bitDepth=ocio.BIT_DEPTH_F16,
        #allocation=ocio.ALLOCATION_LG2,
        #allocationVars=[-15, 6]
    )

    sRGB_inv_eotf = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_SCENE,
        name="sRGB (gamma ~2.2)",
        description="sRGB inverse EOTF",
        encoding='sdr-video',
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.ExponentWithLinearTransform(gamma=[2.4, 2.4, 2.4, 1.],
                                             offset=[0.055, 0.055, 0.055, 0.0],
                                             direction=ocio.TRANSFORM_DIR_INVERSE),

    )

    sRGB = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_SCENE,
        name="sRGB",
        description="sRGB inverse EOTF",
        encoding='sdr-video',
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform([
            ocio.ExponentTransform(value=[2.2, 2.2, 2.2, 1.], direction=ocio.TRANSFORM_DIR_INVERSE)
        ]),
    )
    data = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_SCENE,
        name='data',
        encoding='data',
        isData=True,
    )

    all_colorspaces = [data, linear, sRGB]

    cfg = ocio.Config()
    _ = [cfg.addColorSpace(cs) for cs in all_colorspaces]
    cfg.addColorSpace(linear)
    cfg.addColorSpace(sRGB)

    #cfg.setRole('aces_interchange', aces.getName())
    #cfg.setRole('cie_xyz_d65_interchange', xyzd65.getName())
    cfg.setRole(ocio.ROLE_DATA, data.getName())
    cfg.setRole(ocio.ROLE_DEFAULT, linear.getName())
    cfg.setRole(ocio.ROLE_REFERENCE, linear.getName())
    cfg.setRole(ocio.ROLE_SCENE_LINEAR, linear.getName())
    cfg.setRole(ocio.ROLE_COLOR_PICKING, sRGB.getName())

    # add default colorimetry view transform to config
    # cfg.addViewTransform(
    #     ocio.ViewTransform(
    #         name='colorimetry',
    #         referenceSpace=ocio.REFERENCE_SPACE_SCENE,
    #         fromReference=ocio.BuiltinTransform('ACES-AP0_to_CIE-XYZ-D65_BFD'))
    #     )
    # )

    default_display_cs_name = 'sRGB'
    cfg.addDisplayView(display='sRGB-like Commodity',
                       view='Inverse EOTF',
                       colorSpaceName=default_display_cs_name)
    #cfg.addSharedView('Colorimetry', 'colorimetry', '<USE_DISPLAY_NAME>')
    #cfg.addDisplaySharedView(default_display, 'Colorimetry')
    return cfg
