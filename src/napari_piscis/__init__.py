from importlib.resources import files
napari_yaml = str(files(__package__) / ".." / ".." / "napari.yaml")