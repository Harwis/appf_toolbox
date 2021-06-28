def fuse_fx17_to_fx10(data_path, data_name_fx17, data_name_fx10,
                      search_radius = 80,
                      batch_size = 100,
                      flag_check=False,
                      check_row=800, check_col=300,
                      flag_save=False):
    # Import tools
    import spectral.io.envi as envi
    from matplotlib import pyplot as plt
    import numpy as np
    from datetime import datetime
    from sklearn.externals import joblib
    from skimage.util import compare_images

    start = datetime.now()

    # LiDar data is corresponding to the last 8, 6 and 4 channel in the hypercube.
    x_ind_lida = -8
    y_ind_lida = -6
    z_ind_lida = -4

    ####################################################################################################################
    # Read meta data
    ####################################################################################################################
    meta_data_fx10 = envi.open(data_path + '/' + data_name_fx10 + '.hdr',
                               data_path + '/' + data_name_fx10 + '.raw')

    meta_data_fx17 = envi.open(data_path + '/' + data_name_fx17 + '.hdr',
                               data_path + '/' + data_name_fx17 + '.raw')

    wave_fx10 = meta_data_fx10.metadata['wavelength']
    wave_fx17 = meta_data_fx17.metadata['wavelength']

    # The last value of wavelength is '', remove it.
    if wave_fx10[-1] == '':
        # wave[-1] = '', remove it.
        del wave_fx10[-1]
    wave_fx10 = np.asarray(wave_fx10, dtype=float)

    if wave_fx17[-1] == '':
        # wave[-1] = '', remove it.
        del wave_fx17[-1]
    wave_fx17 = np.asarray(wave_fx17, dtype=float)

    print('Data type of FX10: ', meta_data_fx10.__class__)
    print('Meta data of FX10: ', meta_data_fx10)
    print('Wavelength of FX10: ', wave_fx10)
    print('Data type of FX17: ', meta_data_fx17.__class__)
    print('Meta data of FX10: ', meta_data_fx17)
    print('Wavelength of FX17: ', wave_fx17)


    ####################################################################################################################
    # Batch processing
    ####################################################################################################################
    # Organise batches
    n_batches = int(meta_data_fx10.nrows / batch_size)
    for batch_n in range(0, n_batches + 1):
        if batch_n == n_batches:
            sta_line = batch_size * n_batches
            end_line = meta_data_fx10.nrows - 1
        else:
            sta_line = batch_size * batch_n
            end_line = batch_size * (batch_n + 1) - 1

        # Load data to memory
        hyp_cube_fx10 = meta_data_fx10.read_subregion((sta_line, end_line + 1), (0, meta_data_fx10.ncols))
        hyp_cube_fx17 = print(meta_data_fx17.read_subimage.__doc__)

        print('Line %d to line %d finished. A total of %.2f percent done.'
              % (sta_line, end_line, (end_line + 1) * 100 / meta_data_fx10.nrows))





    stop = datetime.now()
    print('Total time used: ', stop - start)

if __name__ == "__main__":
    data_path = '/media/huajian/Files/Data/FE_night_trial'
    data_name_fx10 = 'Combined_SpecimFX10_TPA_SILO_1_20210518-003.cmb'
    data_name_fx17 = 'Combined_SpecimFX17_TPA_SILO_1_20210518-003.cmb'

    fuse_fx17_to_fx10(data_path, data_name_fx17, data_name_fx10,
                      search_radius=80,
                      batch_size=100,
                      flag_check=False,
                      check_row=800, check_col=300,
                      flag_save=False)