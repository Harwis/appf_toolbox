def check_fx10_dark_error(data_path, data_name, threshold=500):
    """
    Check dark reference errors of the FX10 camera in the WIWAM system.
    :param data_path: The path of the hyperspectral imaging data
    :param data_name: The name of the data
    :param threshold: The threshold to determine if there are errors or not. Default is 500
    :return: If it has errors, it will return 1; otherwise return 0.
    """

    import spectral.io.envi as envi

    # Load data
    meta_data = envi.open(data_path + '/' + data_name + '/capture/' + 'DARKREF_' + data_name + '.hdr',
                          data_path + '/' + data_name + '/capture/' + 'DARKREF_' + data_name + '.raw')

    data = meta_data.load()

    # If any of the raw values is bigger than the threshold, then the data has errors.
    max = data.max()

    if max > threshold:
        return 1
    else:
        return 0
