def test_import_and_widget():
    import napari_piscis._widget as w
    assert callable(w.make_inference_widget)