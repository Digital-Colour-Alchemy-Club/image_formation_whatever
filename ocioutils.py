import PyOpenColorIO
import colour
import PyOpenColorIO as ocio
import numpy
import numpy as np

from aenum import Enum


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
    GAMMA10 = "Gamma 1.0"
    PFWL = "ARRI Gamma ~2.4"
    PFWL26 = "ARRI Gamma ~2.6"
    ADOBE_1998 = "Adobe (1998)"

    @property
    def decoding_xform(self):
        cls = DisplayCCTFs
        return {
            cls.L_STAR: display_cctf(gamma=3.0, offset=0.16),
            cls.SRGB: display_cctf(gamma=2.4, offset=0.055),
            cls.REC709: display_cctf(gamma=1 / 0.45, offset=0.099),
            cls.GAMMA26: display_cctf(gamma=2.6),
            cls.GAMMA24: display_cctf(gamma=2.4),
            cls.GAMMA22: display_cctf(gamma=2.2),
            cls.GAMMA18: display_cctf(gamma=1.8),
            cls.PFWL: display_cctf(gamma=2.725, offset=0.097),
            cls.PFWL26: display_cctf(gamma=2.9961, offset=0.1111),
            cls.ADOBE_1998: display_cctf(gamma=563 / 256.0),
        }[self]

    @property
    def encoding_xform(self):
        xf = self.decoding_xform
        xf.setDirection(1)
        return xf


DISPLAY_BUILTINS = {
    Displays.REC_709: "CIE-XYZ-D65_to_Display-REC709",
    Displays.REC_2020: "CIE-XYZ-D65_to_Display-REC2020",
    Displays.SRGB: "CIE-XYZ-D65_to_Display-sRGB",
    Displays.P3_DCI: "CIE-XYZ-D65_to_Display-P3DCI-BFD",
    Displays.P3_D60: "CIE-XYZ-D65_to_Display-P3D60-BFD",
    Displays.P3_D65: "CIE-XYZ-D65_to_Display-P3D65",
    Displays.DCDM: "CIE-XYZ-D65_to_Display-DCDM",
    Displays.REC_2100_PQ: "CIE-XYZ-D65_to_Display-REC2100-PQ",
    Displays.REC_2100_HLG: "CIE-XYZ-D65_to_Display-REC2100-HLG",
    Displays.ST_2084_P3_D65: "CIE-XYZ-D65_to_Display-ST2084-P3D65",
}


def aces_viewtransform_factory(
    display=Displays.SRGB,
    d60sim=False,
    legal=False,
    limit=None,
):
    return NotImplemented


# helpers


def collapse_redundant_channels(table, max_channels=3):
    t = np.copy(table)
    t *= np.ones(max_channels)
    if np.allclose(*[t[..., c] for c in range(max_channels)]):
        return t[..., 0]
    return t


def _rgba_tuple(x, alpha=1.0):
    return [x] * 3 + [float(alpha)]


def get_name(obj):
    """Retrieve the name of an OCIO object"""
    if hasattr(obj, "getName"):
        return obj.getName()
    elif hasattr(obj, "name"):
        return obj.name
    else:
        return obj


# transform helpers
def display_cctf(gamma=1.0, offset=None, direction=None, negative_style=None):
    if offset:
        cctf = ocio.ExponentWithLinearTransform()
        cctf.setGamma(_rgba_tuple(gamma))
        cctf.setOffset(_rgba_tuple(offset, alpha=0.0))
    else:
        cctf = ocio.ExponentTransform(_rgba_tuple(gamma))

    if direction:
        cctf.setDirection(direction)
    if negative_style:
        cctf.setNegativeStyle(negative_style)

    return cctf


def _simplify_transform(
    transform,
    optimization=ocio.OPTIMIZATION_VERY_GOOD,
    direction=ocio.TRANSFORM_DIR_FORWARD,
    config=None,
    context=None,
):
    config = config or ocio.GetCurrentConfig()
    context = context or config.getCurrentContext()
    proc = config.getProcessor(
        transform=transform, context=context, direction=direction
    )
    gt = proc.getOptimizedProcessor(optimization).createGroupTransform()
    return gt[0] if len(gt) < 2 else gt


def _convert_numpy_matrix33_to_matrix_transform(
    matrix33, direction=ocio.TRANSFORM_DIR_FORWARD
):
    mtx = np.pad(np.array(matrix33).flatten().reshape(3, 3), [(0, 1), (0, 1)]).flatten()
    mtx[-1] = 1
    return ocio.MatrixTransform(matrix=mtx, direction=direction)


def _convert_matrix_transform_to_numpy_matrix33(
    matrix_transform, direction=ocio.TRANSFORM_DIR_FORWARD
):
    mt = _simplify_transform(matrix_transform, direction=direction)
    return np.array(mt.getMatrix()).reshape(4, 4)[:3, :3]


def get_camera_colorspaces(family=""):
    colorspaces = []
    # get Camera Log colorspaces from Builtins
    for name, description in ocio.BuiltinTransformRegistry().getBuiltins():
        if name.endswith("to_ACES2065-1"):
            cs = ocio.ColorSpace(
                name=description.split("Convert ")[-1].split(" to ")[0],
                family=family,
                encoding="log",
                toReference=ocio.BuiltinTransform(name),
            )
            colorspaces.append(cs)
    return colorspaces


def get_display_colorspaces(family=""):
    colour.RGB_COLOURSPACES["CIE-XYZ (D65)"] = colour.RGB_Colourspace(
        name="CIE-XYZ (D65)",
        primaries=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        whitepoint=np.array([0.3127, 0.329]),
        whitepoint_name="D65",
        use_derived_matrix_RGB_to_XYZ=True,
        use_derived_matrix_XYZ_to_RGB=True,
        cctf_decoding=colour.models.linear_function,
        cctf_encoding=colour.models.linear_function,
    )

    xyzd65_to_rec709 = _simplify_transform(
        ocio.GroupTransform(
            [
                ocio.BuiltinTransform(
                    "ACES-AP1_to_CIE-XYZ-D65_BFD", ocio.TRANSFORM_DIR_INVERSE
                ),
                ocio.BuiltinTransform("ACES-AP1_to_LINEAR-REC709_BFD"),
            ]
        )
    )

    # xyzd65_to_rec709 = _convert_numpy_matrix33_to_matrix_transform(
    #     colour.matrix_RGB_to_RGB(
    #         input_colourspace=colour.RGB_COLOURSPACES['CIE-XYZ (D65)'],
    #         output_colourspace=colour.RGB_COLOURSPACES['ITU-R BT.709'],
    #         chromatic_adaptation_transform=None
    #     )
    # )

    xyzd65_to_p3d65 = _convert_numpy_matrix33_to_matrix_transform(
        colour.matrix_RGB_to_RGB(
            input_colourspace=colour.RGB_COLOURSPACES["CIE-XYZ (D65)"],
            output_colourspace=colour.RGB_COLOURSPACES["P3-D65"],
            chromatic_adaptation_transform=None,
        )
    )

    xyzd65_to_p3 = _convert_numpy_matrix33_to_matrix_transform(
        colour.matrix_RGB_to_RGB(
            input_colourspace=colour.RGB_COLOURSPACES["CIE-XYZ (D65)"],
            output_colourspace=colour.RGB_COLOURSPACES["DCI-P3"],
            chromatic_adaptation_transform="Bradford",
        )
    )

    # check
    xyzd65_to_p3_d60_sim_bfd = ocio.GroupTransform(
        [
            ocio.BuiltinTransform(
                "ACES-AP0_to_CIE-XYZ-D65_BFD", ocio.TRANSFORM_DIR_INVERSE
            ),
            _convert_numpy_matrix33_to_matrix_transform(
                colour.matrix_RGB_to_RGB(
                    input_colourspace=colour.RGB_COLOURSPACES["ACES2065-1"],
                    output_colourspace=colour.RGB_COLOURSPACES["DCI-P3"],
                    chromatic_adaptation_transform="Bradford",
                )
            ),
        ]
    )

    # Shapers
    acescct_cctf = ocio.BuiltinTransform("ACEScct-LOG_to_LIN")

    sRGB_g22 = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        name="sRGB (Gamma ~2.2)",
        description="sRGB (Gamma ~2.2) with Rec.709 primaries",
        encoding="sdr-video",
        family=family,
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform(
            [
                xyzd65_to_rec709,
                display_cctf(
                    gamma=2.4, offset=0.055, direction=ocio.TRANSFORM_DIR_INVERSE
                ),
                ocio.RangeTransform(
                    minInValue=0, minOutValue=0, maxInValue=1, maxOutValue=1
                ),
            ]
        ),
    )

    sRGB = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        name="sRGB",
        description="sRGB with Rec.709 primaries",
        encoding="sdr-video",
        family=family,
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform(
            [
                xyzd65_to_rec709,
                display_cctf(gamma=2.2, direction=ocio.TRANSFORM_DIR_INVERSE),
                ocio.RangeTransform(
                    minInValue=0, minOutValue=0, maxInValue=1, maxOutValue=1
                ),
            ]
        ),
    )
    Rec709_display = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        name="Rec.709 (legacy)",
        description="ITU-R BT.709 inverse camera CCTF with BT.709 primaries",
        encoding="sdr-video",
        family=family,
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform(
            [
                xyzd65_to_rec709,
                display_cctf(
                    gamma=1 / 0.45, offset=0.099, direction=ocio.TRANSFORM_DIR_INVERSE
                ),
                ocio.RangeTransform(
                    minInValue=0, minOutValue=0, maxInValue=1, maxOutValue=1
                ),
            ]
        ),
    )

    Rec1886 = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        name="Rec.1886",
        description="ITU-R BT.1886 with BT.709 primaries and D65 whitepoint",
        encoding="sdr-video",
        family=family,
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform(
            [
                xyzd65_to_rec709,
                display_cctf(gamma=2.4, direction=ocio.TRANSFORM_DIR_INVERSE),
                ocio.RangeTransform(
                    minInValue=0, minOutValue=0, maxInValue=1, maxOutValue=1
                ),
            ]
        ),
    )

    DCI_P3 = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        name="DCI-P3",
        description="Gamma 2.6 with DCI primaries",
        encoding="sdr-video",
        family=family,
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform(
            [
                xyzd65_to_p3,
                display_cctf(gamma=2.6, direction=ocio.TRANSFORM_DIR_INVERSE),
                ocio.RangeTransform(
                    minInValue=0, minOutValue=0, maxInValue=1, maxOutValue=1
                ),
            ]
        ),
    )

    P3_D65 = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        name="P3-D65",
        description="Gamma 2.6 with DCI primaries and D65 whitepoint",
        encoding="sdr-video",
        family=family,
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform(
            [
                xyzd65_to_p3d65,
                display_cctf(gamma=2.6, direction=ocio.TRANSFORM_DIR_INVERSE),
                ocio.RangeTransform(
                    minInValue=0, minOutValue=0, maxInValue=1, maxOutValue=1
                ),
            ]
        ),
    )
    # TODO: D60 sim; hdr colorspaces

    return [
        sRGB,
        Rec1886,
        DCI_P3,
        P3_D65,
    ]


def _add_default_colorspaces_and_roles(config=None, scene_linear="ACEScg"):
    # create default colorspaces

    linear_allocation_vars = [-8, 5, 0.00390625]
    # data / no-color
    raw = ocio.ColorSpace(name="raw", isData=True, equalityGroup="nc", encoding="data")

    # Scene and display reference
    aces = ocio.ColorSpace(
        name="ACES2065-1",
        equalityGroup="ap0",
        encoding="scene-linear",
        allocation=ocio.ALLOCATION_LG2,
        allocationVars=linear_allocation_vars,
    )

    xyzd65 = ocio.ColorSpace(
        name="CIE-XYZ (D65)",
        equalityGroup="xyz",
        encoding="display-linear",
        referenceSpace=ocio.REFERENCE_SPACE_DISPLAY,
        allocation=ocio.ALLOCATION_LG2,
        allocationVars=linear_allocation_vars,
    )

    # Other linear reference spaces
    acescg = ocio.ColorSpace(
        name="ACEScg",
        equalityGroup="ap1",
        toReference=ocio.BuiltinTransform("ACEScg_to_ACES2065-1"),
        encoding="scene-linear",
        allocation=ocio.ALLOCATION_LG2,
        allocationVars=linear_allocation_vars,
    )

    lin709 = ocio.ColorSpace(
        name="Linear-Rec709",
        encoding="scene-linear",
        toReference=ocio.GroupTransform(
            [
                ocio.BuiltinTransform(
                    "ACES-AP1_to_LINEAR-REC709_BFD",
                    direction=ocio.TRANSFORM_DIR_INVERSE,
                ),
                ocio.BuiltinTransform("ACEScg_to_ACES2065-1"),
            ]
        ),
        allocation=ocio.ALLOCATION_LG2,
        allocationVars=linear_allocation_vars,
    )

    cfg = config or ocio.GetCurrentConfig()

    # add default colorspaces to config and update roles
    _ = [cfg.addColorSpace(cs) for cs in [raw, aces, xyzd65, acescg, lin709]]

    scene_linear = {acescg.getName(): acescg, lin709.getName(): lin709}[scene_linear]

    cfg.setRole("aces_interchange", aces.getName())
    cfg.setRole("cie_xyz_d65_interchange", xyzd65.getName())
    cfg.setRole(ocio.ROLE_DATA, raw.getName())
    cfg.setRole(ocio.ROLE_DEFAULT, scene_linear.getName())
    cfg.setRole(ocio.ROLE_REFERENCE, aces.getName())
    cfg.setRole(ocio.ROLE_SCENE_LINEAR, scene_linear.getName())

    # add default colorimetry view transform to config
    cfg.addViewTransform(
        ocio.ViewTransform(
            name="colorimetry",
            referenceSpace=ocio.REFERENCE_SPACE_SCENE,
            fromReference=ocio.BuiltinTransform("ACES-AP0_to_CIE-XYZ-D65_BFD"),
        )
    )
    return cfg


# @st.cache(hash_funcs={ocio.Config: id})
def create_new_config(
    scene_linear="ACEScg", default_display=Displays.SRGB, config=None
):
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

    if hasattr(default_display, "value"):
        default_display = default_display.value
    cfg.addDisplayView(default_display, "Colorimetry", "Colorimetry")
    # cfg.addSharedView('Colorimetry', 'colorimetry', '<USE_DISPLAY_NAME>')
    # cfg.addDisplaySharedView(default_display.value, 'Colorimetry')
    return cfg


def serialize_processor(processor, format="Color Transform Format", filename=None):
    if (
        "clf" in str(format).lower()
        or str(format).lower().startswith("academy")
        or str(filename).lower().endswith("clf")
    ):
        fmt = "Academy/ASC Common LUT Format"
    if filename:
        # TODO: return CLF stuff
        # TODO: use GroupTransform.write interface
        processor.write(fmt, filename)
    return processor.write(fmt)


def ocio_colourspace_process(
    a, config, source, target=None, display=None, view=None, use_cpu=True
):
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
        if use_cpu
        else processor.getDefaultGPUProcessor()
    )

    image_desc = ocio.PackedImageDesc(a, width, height, channels)

    processor.apply(image_desc)

    return image_desc.getData().reshape([height, width, channels])


def apply_ocio_processor(processor, RGB, use_gpu=False):
    a = colour.utilities.as_float_array(RGB, dtype=np.float32)
    height, width, channels = a.shape
    image_desc = ocio.PackedImageDesc(a, width, height, channels)

    processor = (
        processor.getDefaultGPUProcessor()
        if use_gpu
        else processor.getDefaultCPUProcessor()
    )

    processor.apply(image_desc)

    return image_desc.getData().reshape([height, width, channels])


def baby_config():

    linear = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_SCENE,
        name="linear",
        description="scene-linear",
        encoding="scene-linear",
        bitDepth=ocio.BIT_DEPTH_F16,
        # allocation=ocio.ALLOCATION_LG2,
        # allocationVars=[-15, 6]
    )
    sRGB_inv_eotf = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_SCENE,
        name="sRGB (gamma ~2.2)",
        description="sRGB inverse EOTF",
        encoding="sdr-video",
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform(
            [
                ocio.ExponentWithLinearTransform(
                    gamma=[2.4, 2.4, 2.4, 1.0],
                    offset=[0.055, 0.055, 0.055, 0.0],
                    direction=ocio.TRANSFORM_DIR_INVERSE,
                ),
                ocio.RangeTransform(
                    minInValue=0, minOutValue=0, maxInValue=1, maxOutValue=1
                ),
            ]
        ),
    )

    sRGB = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_SCENE,
        name="sRGB",
        description="sRGB inverse EOTF",
        encoding="sdr-video",
        bitDepth=ocio.BIT_DEPTH_UINT10,
        fromReference=ocio.GroupTransform(
            [
                ocio.ExponentTransform(
                    value=[2.2, 2.2, 2.2, 1.0], direction=ocio.TRANSFORM_DIR_INVERSE
                ),
                ocio.RangeTransform(
                    minInValue=0, minOutValue=0, maxInValue=1, maxOutValue=1
                ),
            ]
        ),
    )
    data = ocio.ColorSpace(
        referenceSpace=ocio.REFERENCE_SPACE_SCENE,
        name="non-colour data",
        encoding="data",
        isData=True,
    )

    all_colorspaces = [data, linear, sRGB]

    cfg = ocio.Config()
    _ = [cfg.addColorSpace(cs) for cs in all_colorspaces]
    cfg.addColorSpace(linear)
    cfg.addColorSpace(sRGB)

    # Consistently throws problems here likely in relation to the above
    # 412-417 lines being called first and this tripping errors in this
    # block. Commenting seems to quiet the issues.
    # cfg.setRole('aces_interchange', aces.getName())
    # cfg.setRole('cie_xyz_d65_interchange', xyzd65.getName())

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

    default_display_cs_name = "sRGB"
    cfg.addDisplayView(
        display="sRGB-like Commodity",
        view="Inverse EOTF",
        colorSpaceName=default_display_cs_name,
    )
    # cfg.addSharedView('Colorimetry', 'colorimetry', '<USE_DISPLAY_NAME>')
    # cfg.addDisplaySharedView(default_display, 'Colorimetry')
    return cfg


def ocio_viewer(
    RGB,
    source=None,
    display=None,
    view=None,
    exposure=0.0,
    contrast=1.0,
    gamma=1.0,
    style=ocio.EXPOSURE_CONTRAST_LINEAR,
    use_gpu=False,
    config=None,
    context=None,
):

    RGB_out = np.copy(RGB)
    cfg = config or create_new_config()
    context = context or cfg.getCurrentContext()

    exposure_contrast_transform = ocio.ExposureContrastTransform(
        exposure=exposure, gamma=gamma, contrast=contrast, style=style
    )

    proc = cfg.getProcessor(
        context=context,
        direction=ocio.TRANSFORM_DIR_FORWARD,
        transform=ocio.GroupTransform(
            [
                exposure_contrast_transform,
                ocio.DisplayViewTransform(
                    src=source or "scene_linear",
                    display=display or cfg.getDefaultDisplay(),
                    view=view or cfg.getDefaultView(),
                ),
            ]
        ),
    )

    RGB_out = apply_ocio_processor(proc, RGB_out, use_gpu=use_gpu)

    return RGB_out


def add_aesthetic_transfer_function_to_config(atf, config):
    min_exposure = -6.5
    max_exposure = atf.ev_above_middle_grey
    middle_grey = atf.middle_grey_in

    domain = np.array([min_exposure, max_exposure])

    lin_to_normalized_log_transform = (
        ocio.AllocationTransform(
            vars=np.log2(middle_grey * np.power(2.0, domain)),
            allocation=ocio.ALLOCATION_LG2,
        ),
    )

    normalized_log_to_lin_transform = (
        ocio.AllocationTransform(
            vars=np.log2(middle_grey * np.power(2.0, domain)),
            allocation=ocio.ALLOCATION_LG2,
            direction=ocio.TRANSFORM_DIR_INVERSE,
        ),
    )

    image_formation_transform = ocio.ViewTransform(
        name="Image Formation Transform",
        fromReference=ocio.GroupTransform(
            [
                ocio.AllocationTransform(
                    vars=np.log2(middle_grey * np.power(2.0, domain)),
                    allocation=ocio.ALLOCATION_LG2,
                ),
                ocio.FileTransform(
                    src=atf.get_filename(extension="clf"),
                    interpolation=ocio.INTERP_TETRAHEDRAL,
                ),
            ]
        ),
    )

    view_transforms = [
        image_formation_transform,
    ]

    for vt in view_transforms:
        config.addViewTransform(vt)

    return config
