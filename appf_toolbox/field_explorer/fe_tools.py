from numba import jit, cuda

@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1

@cuda.jit
def fuse_subregion_fx17_to_fx10(data_path,
                                data_name_fx17,
                                data_name_fx10,
                                sta_line_fx10,
                                end_line_fx10,
                                search_radius=80,
                                x_ind_lida=-8,
                                y_ind_lida=-6,
                                z_ind_lida=-4,
                                flag_check=False,
                                check_row=10,
                                check_col=50,
                                flag_print_progress=True,
                                flag_save=False):

    """
    Fuse a subregion of FX17 data to FX10 data of the FieldExplore.
    Authour: Huajian Liu  huajian.liu@adelaide.edu.au
    Version: 1.0 Date: 01 July, 2021

    :param data_path: The path of the data.
    :param data_name_fx17: The name of FX17 data.
    :param data_name_fx10: The name of FX10 data.
    :param sta_line_fx10: Starting line for processing in FX10 data. Counted from 0.
    :param end_line_fx10: Ending line for processing in FX10 data. Counted from 0.
    :param search_radius: The search wind radius in the FX17 data. This is used for speed up processing. Default is 80.
    :param x_ind_lida: The x index of LiDar value. Default is -8. Indies of xyz is (-8, -6, -4) or (-9, -7, -5).
    :param y_ind_lida: The y index of LiDar value. Default is -6.
    :param z_ind_lida: The z index of LiDar value. Default is -5.
    :param flag_check: If set this flag to True, it will show a figure of fusion result
    :param check_row: The row number for checking in FX10. Default is 0.
    :param check_col: The col number for checking in FX10. Default is 0.
    :param flag_print_progress: If set this flag to True, it will print the processing progress but slow down the speed.
    :param flag_save: If set this flag to True, the fused data will be saved in the current working folder as .sav file.
    :return: A dictionary containing the fused data and the meta data of the original FX10 and FX17 data.
    """

    # Import tools
    import spectral.io.envi as envi
    from matplotlib import pyplot as plt
    import numpy as np
    from datetime import datetime
    from skimage.util import compare_images

    start_time = datetime.now()

    print('------------------------------------------------------')
    print('Processing line %d to line %d.' % (sta_line_fx10, end_line_fx10))
    print('------------------------------------------------------')

    # ------------------------------------------------------------------------------------------------------------------
    # Read meta data
    # ------------------------------------------------------------------------------------------------------------------
    meta_data_fx10 = envi.open(data_path + '/' + data_name_fx10 + '.hdr',
                               data_path + '/' + data_name_fx10 + '.raw')

    meta_data_fx17 = envi.open(data_path + '/' + data_name_fx17 + '.hdr',
                               data_path + '/' + data_name_fx17 + '.raw')

    if flag_print_progress:
        print('Data type of FX10: ', meta_data_fx10.__class__)
        print('Meta data of FX10: ', meta_data_fx10)
        print('Data type of FX17: ', meta_data_fx17.__class__)
        print('Meta data of FX17: ', meta_data_fx17)

    # ------------------------------------------------------------------------------------------------------------------
    # Clear up the wavelengths
    # ------------------------------------------------------------------------------------------------------------------
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

    # If there are overlapped wavelengths in fx10 and fx17 data, remove the overlapped wavelengths of fx17
    for i in range(0, wave_fx17.shape[0]):
        if wave_fx17[i] > np.max(wave_fx10):
            start_wave_ind_fx17 = i
            break

    # The clear wavelengths. The last 11 wavelengths are for LiDar data and need to be removed.
    wave_fx10 = wave_fx10[0 : -11]
    wave_fx17 = wave_fx17[start_wave_ind_fx17: -11]
    fused_wave = np.concatenate((wave_fx10, wave_fx17), axis=0)

    # ------------------------------------------------------------------------------------------------------------------
    # Load the subregion of fx10 data
    # ------------------------------------------------------------------------------------------------------------------
    # Check if the staring line and ending line are legal
    if sta_line_fx10 < 0:
        print('Error: Staring line is smaller than 0.')
        return -1
    elif end_line_fx10 > meta_data_fx10.nrows - 1:
        print('Error: Ending line index is bigger than %d which is the max row index of FX10 data.'
              % meta_data_fx10.nrows)
        return -1
    elif sta_line_fx10 > end_line_fx10:
        print('Error: starting line index is bigger than ending line index. ')
        return -1
    else:
        pass

    # Load hypercube of fx10 from sta_line to end_line to memory.
    hypcube_fx10 = meta_data_fx10.read_subregion((sta_line_fx10, end_line_fx10 + 1),
                                                  (0, meta_data_fx10.ncols),
                                                  (list(range(0, wave_fx10.shape[0]))))

    # Load LiDar data of fx10 from sta_line to end_line
    # First, convert indices of LiDar data to positive numbers
    x_ind_lida_fx10 = meta_data_fx10.nbands + x_ind_lida - 1
    y_ind_lida_fx10 = meta_data_fx10.nbands + y_ind_lida - 1
    z_ind_lida_fx10 = meta_data_fx10.nbands + z_ind_lida - 1
    xyz_fx10 = meta_data_fx10.read_subimage(list(range(sta_line_fx10, end_line_fx10 + 1)),
                                            list(range(0, meta_data_fx10.ncols)),
                                            [x_ind_lida_fx10, y_ind_lida_fx10, z_ind_lida_fx10])

    # ------------------------------------------------------------------------------------------------------------------
    # Fusing
    # ------------------------------------------------------------------------------------------------------------------
    # Make a zero-hypercub for save the transformed data of fx17
    hypcube_fx17_to_fx10 = np.zeros((hypcube_fx10.shape[0], hypcube_fx10.shape[1], wave_fx17.shape[0]))

    # For checking purpose only
    record_global_row_fx17_rough = np.zeros((hypcube_fx10.shape[0], hypcube_fx10.shape[1]))
    record_global_col_fx17_rough = np.zeros((hypcube_fx10.shape[0], hypcube_fx10.shape[1]))
    record_global_row_fx17_accu = np.zeros((hypcube_fx10.shape[0], hypcube_fx10.shape[1]))
    record_global_col_fx17_accu = np.zeros((hypcube_fx10.shape[0], hypcube_fx10.shape[1]))

    # Up sampling FX17 to FX10 spatial resolution
    for local_row_fx10 in range(0, hypcube_fx10.shape[0]): # For each row
        for local_col_fx10 in range(0, hypcube_fx10.shape[1]): # For each pixel in a row
            # ==========================================================================================================
            # For each pixel in fx10, find the rough pixel location in the fx17 using the ratio.
            # ==========================================================================================================
            # Global pixel location of fx10
            global_row_fx10 = sta_line_fx10 + local_row_fx10
            global_col_fx10 = local_col_fx10
            # print('')
            # print('global (row, col) of FX10: ', (global_row_fx10, global_col_fx10))

            # Convert global pixel location of fx10 to ratio
            rat_row = (global_row_fx10 + 1) / meta_data_fx10.nrows
            rat_col = (global_col_fx10 + 1) / meta_data_fx10.ncols

            # Rough corresponding pixel location in FX17
            global_row_fx17_rough = int(np.round(meta_data_fx17.nrows * rat_row) - 1)
            global_col_fx17_rough = int(np.round(meta_data_fx17.ncols * rat_col) - 1)
            # print('Rough (row, col) of FX17: ', (global_row_fx17_rough, global_col_fx17_rough))


            # ==========================================================================================================
            # Based on the rough pixel location in fx17, find the accurate pixel location using the xyz distance.
            # ==========================================================================================================
            # ----------------------------------------------------------------------------------------------------------
            # Define a search window centred in the rough pixel location.
            # ----------------------------------------------------------------------------------------------------------
            win_row_top = global_row_fx17_rough - search_radius
            win_row_bot = global_row_fx17_rough + search_radius
            win_col_lef = global_col_fx17_rough - search_radius
            win_col_rig = global_col_fx17_rough + search_radius

            # When the search window is out of the image, do the following.
            if win_row_top < 0:
                win_row_top = 0
            if win_row_bot > meta_data_fx17.nrows - 1:
                win_row_bot = meta_data_fx17.nrows - 1
            if win_col_lef < 0:
                win_col_lef = 0
            if win_col_rig > meta_data_fx17.ncols - 1:
                win_col_rig = meta_data_fx17.ncols - 1

            # ----------------------------------------------------------------------------------------------------------
            # The distances between between the current xyz of fx10 to the xyz in the window of fx17
            # ----------------------------------------------------------------------------------------------------------
            # xyz in the window of fx17.
            x_ind_lida_fx17 = meta_data_fx17.nbands + x_ind_lida - 1
            y_ind_lida_fx17 = meta_data_fx17.nbands + y_ind_lida - 1
            z_ind_lida_fx17 = meta_data_fx17.nbands + z_ind_lida - 1
            neighbours_xyz_fx17 = meta_data_fx17.read_subimage(list(range(win_row_top, win_row_bot + 1)),
                                                               list(range(win_col_lef, win_col_rig + 1)),
                                                               [x_ind_lida_fx17, y_ind_lida_fx17, z_ind_lida_fx17])

            # Tile the current xyz of fx10 for computing the distance
            current_xyz_fx10 = np.tile(xyz_fx10[local_row_fx10, local_col_fx10].reshape((1, 1, 3)),
                                       (neighbours_xyz_fx17.shape[0], neighbours_xyz_fx17.shape[1], 1))

            dis = np.sum((current_xyz_fx10 - neighbours_xyz_fx17) ** 2, axis=2)

            # ----------------------------------------------------------------------------------------------------------
            # Accurate pixel location in fx17
            # ----------------------------------------------------------------------------------------------------------
            # if np.all(dis == dis[0, 0]) or dis.min() == 0:
            if np.all(xyz_fx10[local_row_fx10, local_col_fx10, :] > 32766):
                # If the current xyz values of FX10 are the same as all of the xyz values in the window of FX17 then
                # set the rough the rough value to the accurate values.
                global_row_col_fx17_accu = np.array([global_row_fx17_rough, global_col_fx17_rough])
                # print('dis same')
                # print('Accurate global (row, col) of FX17:', global_row_col_fx17_accu)

            else:
                # The pixel location in the window of fx17 returning the minimum distance.
                win_row_col_fx17 = np.unravel_index(np.argmin(dis), dis.shape, order='C')

                # The corresponding global (accurate) pixel location of FX17
                global_row_col_fx17_accu = win_row_col_fx17 + np.array([win_row_top, win_col_lef])
                # print('dis not same')
                # print('Accurate global (row, col) of FX17', global_row_col_fx17_accu)

            # The starting line and ending line of fx17 data for processing.
            if global_row_fx10 == sta_line_fx10:
                sta_line_fx17 = global_row_col_fx17_accu[0]
            if global_row_fx10 == end_line_fx10:
                end_line_fx17 = global_row_col_fx17_accu[0]

            # ----------------------------------------------------------------------------------------------------------
            # Copy the reflectance value of FX17 at this pixel to the corresponding pixel of fx10 data.
            # ----------------------------------------------------------------------------------------------------------
            hypcube_fx17_to_fx10[local_row_fx10, local_col_fx10, :] = \
            meta_data_fx17.read_subregion((global_row_col_fx17_accu[0], global_row_col_fx17_accu[0] + 1),
                                          (global_row_col_fx17_accu[1], global_row_col_fx17_accu[1] + 1),
                                          list(range(start_wave_ind_fx17, meta_data_fx17.nbands - 11)))
            
            # Combine hypcube_fx10 and the transformed hypercube from fx17 to fx10.
            fused_hypcube = np.concatenate((hypcube_fx10, hypcube_fx17_to_fx10), axis=2)

            # Record the rough and accurate pixel location of FX17 for checking.
            record_global_row_fx17_rough[local_row_fx10, local_col_fx10] = global_row_fx17_rough
            record_global_col_fx17_rough[local_row_fx10, local_col_fx10] = global_col_fx17_rough
            record_global_row_fx17_accu[local_row_fx10, local_col_fx10] = global_row_col_fx17_accu[0]
            record_global_col_fx17_accu[local_row_fx10, local_col_fx10] = global_row_col_fx17_accu[1]

        if flag_print_progress:
            print('Global line of FX10 %d (local %d ) | (Rough, accurate) global line of FX17 (%d, %d) | %.2f percent Finished. ' %
                  (global_row_fx10,
                   local_row_fx10,
                   global_row_fx17_rough,
                   global_row_col_fx17_accu[0],
                   (local_row_fx10 + 1) * 100 / (end_line_fx10 - sta_line_fx10 + 1)))

    # After finishing the processing, save the data to dictionary.
    dict = {'hypcube': fused_hypcube,
            'xyz': xyz_fx10,
            'wavelength': fused_wave,
            'meta_fx10': meta_data_fx10,
            'meta_fx17': meta_data_fx17,
            'line_range_fx10': (sta_line_fx10, end_line_fx10),
            'line_range_fx17': (sta_line_fx17, end_line_fx17),
            'row_fx17_rough': record_global_row_fx17_rough,
            'col_fx17_rough': record_global_col_fx17_rough,
            'row_fx17_accurate': record_global_col_fx17_accu,
            'col_fx17_accurate': record_global_col_fx17_accu}


    # ------------------------------------------------------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------------------------------------------------------
    if flag_check:
        # Check the last band image of fx10 and the first band image of fx17 in the fused data
        # The original fx17 band image
        img_fx17 = meta_data_fx17.read_subregion((sta_line_fx17, end_line_fx17 + 1),
                                                 (0, meta_data_fx17.ncols),
                                                 [start_wave_ind_fx17])

        # The last band image of fx10 in the fused data
        img_fx10 = fused_hypcube[:, :, wave_fx10.shape[0] - 1]

        # The image transformed from fx17 to fx10
        img_fx17_to_fx10 = fused_hypcube[:, :, wave_fx10.shape[0]]

        # The difference of the images
        diff = compare_images(img_fx10, img_fx17_to_fx10, method='diff')

        fig0, ax_fig0 = plt.subplots(2, 5)
        ax_fig0[0, 0].imshow(img_fx17, cmap='jet')
        ax_fig0[0, 0].set_title('FX17 at ' + str(wave_fx17[0]) + ' nm')
        ax_fig0[0, 1].imshow(img_fx10, cmap='jet')
        ax_fig0[0, 1].plot(check_col, check_row, 'r+')
        ax_fig0[0, 1].set_title('FX10 at ' + str(wave_fx10[-1]) + ' nm')
        ax_fig0[0, 2].imshow(img_fx17_to_fx10, cmap='jet')
        ax_fig0[0, 2].set_title('FX17 transformed to FX10')
        ax_fig0[0, 3].imshow(diff, cmap='jet')
        ax_fig0[0, 3].set_title('Difference')
        ax_fig0[0, 4].imshow(xyz_fx10[:, :, 2], cmap='jet')
        ax_fig0[0, 4].set_title('z of FX10')

        for ax in ax_fig0[1, :]:
            ax.remove()
        gs = ax_fig0[1, 3].get_gridspec()
        ax_fig0_bot = fig0.add_subplot(gs[1, :])
        ax_fig0_bot.plot(fused_wave, fused_hypcube[check_row, check_col, :].reshape((fused_hypcube.shape[2], )))
        ax_fig0_bot.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        ax_fig0_bot.set_ylabel('Reflectance', fontsize=12, fontweight='bold')

    end_time = datetime.now()
    print('Fusion finished. Total time used:', end_time - start_time)

    if flag_save:
        from sklearn.externals import joblib
        joblib.dump(dict, data_name_fx10[-27:-5] + '_line_' + str(sta_line_fx10) + '_to_' + str(end_line_fx10) + '.sav')

    return dict


def fuse_fx17_to_fx10(data_path,
                      data_name_fx17,
                      data_name_fx10,
                      batch_size = 100,
                      search_radius = 80,
                      x_ind_lida = -8,
                      y_ind_lida = -6,
                      z_ind_lida = -4,
                      flag_check=False,
                      check_row=0,
                      check_col=0,
                      flag_print_progress=False,
                      flag_save=False):

    # Import tools
    import spectral.io.envi as envi
    from datetime import datetime
    from sklearn.externals import joblib

    start = datetime.now()

    # Read meta data
    meta_data_fx10 = envi.open(data_path + '/' + data_name_fx10 + '.hdr',
                               data_path + '/' + data_name_fx10 + '.raw')


    # Batch processing
    n_batches = int(meta_data_fx10.nrows / batch_size)

    for batch_n in range(0, n_batches + 1):
        print('')
        print(' Processing batch ' + str(batch_n) + '......')
        if batch_n == n_batches:
            sta_line = batch_size * n_batches
            end_line = meta_data_fx10.nrows - 1
        else:
            sta_line = batch_size * batch_n
            end_line = batch_size * (batch_n + 1) - 1

        # Process a batch
        fused_data = fuse_subregion_fx17_to_fx10(data_path,
                                                 data_name_fx17,
                                                 data_name_fx10,
                                                 sta_line_fx10=sta_line,
                                                 end_line_fx10=end_line,
                                                 search_radius=search_radius,
                                                 x_ind_lida=x_ind_lida,
                                                 y_ind_lida=y_ind_lida,
                                                 z_ind_lida=z_ind_lida,
                                                 flag_check=flag_check,
                                                 check_row=check_row,
                                                 check_col=check_col,
                                                 lag_print_progress=flag_print_progress)

        # Save a batch
        if flag_save:
            joblib.dump(fused_data, data_name_fx10[-27:-5] +
                        '_line_' + str(sta_line) + '_to_' + str(end_line) + '.sav')

    stop = datetime.now()
    print('Total time used: ', stop - start)


########################################################################################################################
# Demostration
########################################################################################################################
# if __name__ == "__main__":
#     data_path = '/media/huajian/Files/Data/FE_night_trial'
#     data_name_fx10 = 'Combined_SpecimFX10_TPA_SILO_1_20210518-003.cmb'
#     data_name_fx17 = 'Combined_SpecimFX17_TPA_SILO_1_20210518-003.cmb'
#
#     # Demo of fuse_subregion_fx17_to_fx10
#     print(fuse_subregion_fx17_to_fx10.__doc__)
#
#     fused_data = fuse_subregion_fx17_to_fx10(data_path,
#                                             data_name_fx17,
#                                             data_name_fx10,
#                                             sta_line_fx10 = 0,
#                                             end_line_fx10 = 199,
#                                             # search_radius=80,
#                                             # x_ind_lida=-8,
#                                             # y_ind_lida=-6,
#                                             # z_ind_lida=-4,
#                                             flag_check=True,
#                                             check_row=10,
#                                             check_col=50,
#                                             flag_print_progress=True,
#                                             flag_save=False)
#
#     from sklearn.externals import joblib
#
#     joblib.dump(fused_data, data_name_fx10[-27:-5] + '_line_' + str(0) + '_to_' + str(99) + '.sav')

increment_a_2D_array([1, 2])