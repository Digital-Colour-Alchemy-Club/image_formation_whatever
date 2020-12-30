import pathlib
import util

LOCAL_DATA = pathlib.Path.cwd() / "data"
EXTERNAL_DEPENDENCIES = {
    # "ACES-1.2 Config": RemoteData(
    #     filename="OpenColorIO-Config-ACES-1.2.zip",
    #     url="https://github.com/colour-science/OpenColorIO-Configs/"
    #       "releases/download/v1.2/OpenColorIO-Config-ACES-1.2.zip",
    #     size=130123781,
    # ),
    "Marcie ACES2065-1": util.RemoteData(
        filename="DigitalLAD.2048x1556.exr",
        url="https://zozobra.s3.us-east-1.amazonaws.com/colour/"
            "images/DigitalLAD.2048x1556.exr",
        size=25518832,
    ),
    "CLF Test Image": util.RemoteData(
        filename="CLF_testImagePrototype_v006.exr",
        url="https://raw.githubusercontent.com/alexfry/CLFTestImage/"
            "master/images/CLF_testImagePrototype_v006.exr",
        size=201549,
    ),
    "Marcie 4K": util.RemoteData(
        filename="marcie-4k.exr",
        url="https://zozobra.s3.us-east-1.amazonaws.com/colour/"
            "images/marcie-4k.exr",
        size=63015668,
    ),
}


def get_dependency(key, local_dir=LOCAL_DATA):
    remote_file = EXTERNAL_DEPENDENCIES[key]
    remote_file.download(output_dir=local_dir)
    return local_dir / remote_file.filename
