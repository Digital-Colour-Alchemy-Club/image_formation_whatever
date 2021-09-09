
def update_imageio():
    # Install imageio freeimage plugin (i.e., for EXR support)
    import imageio
    imageio.plugins.freeimage.download()


def run_bootstrap():
    update_imageio()

