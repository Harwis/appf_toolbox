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


def fix_fx10_dark_error(error_data_path,
                        error_data_list_path, error_data_list_file,
                        good_data_path, good_data_name):
    """
    Fix the fx10 dark reference error.
    :param error_data_path:
    :param error_data_list_path:
    :param error_data_list_file: a .txt file lists the data names of error data
    :param good_data_path:
    :param good_data_name:
    :return: Return True if the errors are fixed; otherwise return False
    """

    import os
    import shutil
    from spectral import envi

    # Read the dark ref of the good data
    meta_good_dark = envi.open(good_data_path + '/' + good_data_name + '/' + 'capture' + '/' + 'DARKREF_' + good_data_name + '.hdr',
                               good_data_path + '/' + good_data_name + '/' + 'capture' + '/' + 'DARKREF_' + good_data_name + '.raw')

    # error data names
    with open(error_data_list_path + '/' + error_data_list_file, 'r') as f:
        lines = f.readlines()

    # Fix errors
    for a_line in lines:
        a_error_data_name = a_line.strip()
        print('Fixing ' + a_error_data_name)

        # Check if the good and error data have the same wavelengths and ncols
        meta_error_dark = envi.open(
            error_data_path + '/' + a_error_data_name + '/' + 'capture' + '/' + 'DARKREF_' + a_error_data_name + '.hdr',
            error_data_path + '/' + a_error_data_name + '/' + 'capture' + '/' + 'DARKREF_' + a_error_data_name + '.raw')

        if meta_good_dark.metadata['wavelength'] != meta_error_dark.metadata['wavelength']:
            print('The wavelengths of good and error data are different!')
            return False
        elif meta_good_dark.metadata['bands'] != meta_error_dark.metadata['bands']:
            print('The number of bands of good and error data are different!')
            return False

        # Fix the errors
        if os.path.isfile(error_data_path + '/' + a_error_data_name + '/capture/DARKREF_' + a_error_data_name + '_ori.hdr'):
            print(a_error_data_name + 'has been already fixed.')
            pass
        else:
            # Fix .hdr file of dark ref
            os.rename(error_data_path + '/' + a_error_data_name + '/capture/DARKREF_' + a_error_data_name + '.hdr',
                  error_data_path + '/' + a_error_data_name + '/capture/DARKREF_' + a_error_data_name + '_ori.hdr')

            shutil.copy(good_data_path + '/' + good_data_name + '/capture/DARKREF_' + good_data_name + '.hdr',
                        error_data_path + '/' + a_error_data_name + '/capture/DARKREF_' + a_error_data_name + '.hdr')

            # Fix .raw file of dark ref
            os.rename(error_data_path + '/' + a_error_data_name + '/capture/DARKREF_' + a_error_data_name + '.raw',
                      error_data_path + '/' + a_error_data_name + '/capture/DARKREF_' + a_error_data_name + '_ori.raw')

            shutil.copy(good_data_path + '/' + good_data_name + '/capture/DARKREF_' + good_data_name + '.raw',
                        error_data_path + '/' + a_error_data_name + '/capture/DARKREF_' + a_error_data_name + '.raw')

    return True


if __name__ == "__main__":
    error_data_path = 'C:/Huajian/data/dark_ref_error'
    error_data_list_path = error_data_path
    error_data_list_file = 'dark_error_list.txt'
    good_data_path = 'C:/Huajian/data/dark_ref_error/good_data'
    good_data_name = 'vnir_100_140_9151_2021-08-24_03-36-30'

    fix_fx10_dark_error(error_data_path,
                        error_data_list_path, error_data_list_file,
                        good_data_path, good_data_name)









