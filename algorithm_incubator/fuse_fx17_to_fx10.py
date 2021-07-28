import sys
import spectral.io.envi as envi
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.externals import joblib
from skimage.util import compare_images

########################################################################################################################
# Parameters
########################################################################################################################
data_path = '/media/huajian/Files/Data/FE_night_trial'
data_name_fx10 = 'Combined_SpecimFX10_TPA_SILO_1_20210518-003.cmb'
data_name_fx17 = 'Combined_SpecimFX17_TPA_SILO_1_20210518-003.cmb'

# The last (8, 6, 4) and (9, 7, 5) channels are different. Do not know which one is the correct x, y, z values.
x_ind_lida = -8 # Corresponding the last 8, 6 and 4 channel in the hypercube.
y_ind_lida = -6
z_ind_lida = -4
# x_ind_lida = -9 # Corresponding the last 9, 7 and 5 channel in the hypercube.
# y_ind_lida = -7
# z_ind_lida = -5

search_radius = 100

# For checking the results
check_row_1 = 885
check_col_1 = 498
check_row_2 = 2232
check_col_2 = 252

flag_save = True
flag_fig = False # Set to True can observe the window but will slow down the processing.


start = datetime.now()
########################################################################################################################
# Load data
########################################################################################################################
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
print('Data type of FX17: ', meta_data_fx17.__class__)
print('Meta data of FX10: ', meta_data_fx17)

hyp_cube_fx10 = meta_data_fx10.load()
hyp_cube_fx17 = meta_data_fx17.load()


########################################################################################################################
# Separate the spectral data and 3D data
########################################################################################################################
lid_fx10 = hyp_cube_fx10[:, :, -11:].copy()
hyp_cube_fx10 = hyp_cube_fx10[:, :, 0:-11]
wave_fx10 = wave_fx10[0:-11]

lid_fx17 = hyp_cube_fx17[:, :, -11:].copy()
hyp_cube_fx17 = hyp_cube_fx17[:, :, 0:-11]
wave_fx17 = wave_fx17[0:-11]

xyz_fx10 = np.concatenate((lid_fx10[:, :, x_ind_lida].reshape((lid_fx10.shape[0], lid_fx10.shape[1], 1)),
                           lid_fx10[:, :, y_ind_lida].reshape((lid_fx10.shape[0], lid_fx10.shape[1], 1)),
                           lid_fx10[:, :, z_ind_lida].reshape((lid_fx10.shape[0], lid_fx10.shape[1], 1))), axis=2)

xyz_fx17 = np.concatenate((lid_fx17[:, :, x_ind_lida].reshape((lid_fx17.shape[0], lid_fx17.shape[1], 1)),
                           lid_fx17[:, :, y_ind_lida].reshape((lid_fx17.shape[0], lid_fx17.shape[1], 1)),
                           lid_fx17[:, :, z_ind_lida].reshape((lid_fx17.shape[0], lid_fx17.shape[1], 1))), axis=2)


########################################################################################################################
# Remove the overlapped bands
# FX10 <= 1000 nm; FX17 > 1000 nm
########################################################################################################################
band_mask_fx10 = wave_fx10 < 1000
wave_fx10 = wave_fx10[band_mask_fx10]
hyp_cube_fx10 = hyp_cube_fx10[:, :, band_mask_fx10]

band_mask_fx17 = wave_fx17 > 1000
wave_fx17 = wave_fx17[band_mask_fx17]
hyp_cube_fx17 = hyp_cube_fx17[:, :, band_mask_fx17]


########################################################################################################################
# Show the search window
# Turning the fig on will slow down the processing
########################################################################################################################
if flag_fig:
    # Plot spectral data
    f0, ax_f0 = plt.subplots(1, 2)
    f0.suptitle('Spectral data')
    ax_f0[0].imshow(hyp_cube_fx10[:, :, -1], cmap='jet')
    ax_f0[0].set_title('FX10' + ' at ' + str(wave_fx10[-1]) + ' nm')
    ax_f0[1].imshow(hyp_cube_fx17[:, :, 0], cmap='jet')
    ax_f0[1].set_title('FX17' + ' at ' + str(wave_fx17[0]) + ' nm')
    plt.pause(0.001)


########################################################################################################################
# Start fusing
########################################################################################################################
# Make a zero-hypercub for save the results of resampling
hyp_cube_fx17_to_fx10 = np.zeros((hyp_cube_fx10.shape[0], hyp_cube_fx10.shape[1], hyp_cube_fx17.shape[2]))

# For checking purpose only
record_row_fx17_rough = np.zeros((hyp_cube_fx10.shape[0], hyp_cube_fx10.shape[1]))
record_col_fx17_rough = np.zeros((hyp_cube_fx10.shape[0], hyp_cube_fx10.shape[1]))
record_row_fx17_accu = np.zeros((hyp_cube_fx10.shape[0], hyp_cube_fx10.shape[1]))
record_col_fx17_accu = np.zeros((hyp_cube_fx10.shape[0], hyp_cube_fx10.shape[1]))

# Up sampling FX17 to FX10 resolution
for row_fx10 in range(0, hyp_cube_fx10.shape[0]):
    for col_fx10 in range(0, hyp_cube_fx10.shape[1]):
        # Convert pixel location of fx10 to ratio
        rat_row = (row_fx10 + 1) / hyp_cube_fx10.shape[0]
        rat_col = (col_fx10 + 1) / hyp_cube_fx10.shape[1]

        # Rough corresponding pixel location in FX17
        row_fx17_rough = int(hyp_cube_fx17.shape[0] * rat_row)
        col_fx17_rough = int(hyp_cube_fx17.shape[1] * rat_col)

        # Define a search window.
        win_row_top = row_fx17_rough - search_radius
        win_row_bot = row_fx17_rough + search_radius
        win_col_lef = col_fx17_rough - search_radius
        win_col_rig = col_fx17_rough + search_radius

        # When the search window is out of the image, do the following.
        if win_row_top < 0:
            win_row_top = 0
        if win_row_bot > hyp_cube_fx17.shape[0] - 1:
            win_row_bot = hyp_cube_fx17.shape[0] - 1
        if win_col_lef < 0:
            win_col_lef = 0
        if win_col_rig > hyp_cube_fx17.shape[1] - 1:
            win_col_rig = hyp_cube_fx17.shape[1] - 1

        if flag_fig:
            # Draw the search window to check
            sca_ax_f0_0 = ax_f0[0].scatter(col_fx10, row_fx10, s=100, marker='+', c='red')
            sca_ax_f0_1_rough = ax_f0[1].scatter(col_fx17_rough, row_fx17_rough, s=100, marker='+', c='red')
            line1 = ax_f0[1].hlines(y=win_row_top, xmin=win_col_lef, xmax=win_col_rig, color='red')
            line2 = ax_f0[1].hlines(y=win_row_bot, xmin=win_col_lef, xmax=win_col_rig, color='red')
            line3 = ax_f0[1].vlines(x=win_col_lef, ymin=win_row_top, ymax=win_row_bot, color='red')
            line4 = ax_f0[1].vlines(x=win_col_rig, ymin=win_row_top, ymax=win_row_bot, color='red')

        # Euclidian distance
        # neighbours_xyz_fx17 = xyz_fx17[win_row_top: win_row_bot, win_col_lef: win_col_rig, :]
        # current_xyz_fx10 = np.tile(xyz_fx10[row_fx10, col_fx10].reshape((1, 1, 3)),
        #                            (neighbours_xyz_fx17.shape[0], neighbours_xyz_fx17.shape[1], 1))

        neighbours_xyz_fx17 = xyz_fx17[win_row_top: win_row_bot, win_col_lef: win_col_rig, 0:2]
        current_xyz_fx10 = np.tile(xyz_fx10[row_fx10, col_fx10, 0 : 2].reshape((1, 1, 2)),
                                   (neighbours_xyz_fx17.shape[0], neighbours_xyz_fx17.shape[1], 1))

        dis = np.sum((current_xyz_fx10 - neighbours_xyz_fx17) ** 2, axis=2)

        # Target pixel location in FX17 is the pixel return the min distance
        # Target location in the window
        row_col_fx17_target = np.unravel_index(np.argmin(dis), dis.shape, order='C')

        # Target location in the pix location of FX17
        row_col_fx17_target = row_col_fx17_target + np.array([win_row_top, win_col_lef])

        if flag_fig:
            sca_ax_f0_1_tar = ax_f0[1].scatter(row_col_fx17_target[1], row_col_fx17_target[0], s=100, marker='.',
                                               c='yellow')
            plt.pause(0.001)
            sca_ax_f0_0.remove()
            sca_ax_f0_1_rough.remove()
            sca_ax_f0_1_tar.remove()
            line1.remove()
            line2.remove()
            line3.remove()
            line4.remove()

        # Copy the reflectance value of FX17 to fused data.
        hyp_cube_fx17_to_fx10[row_fx10, col_fx10, :] = hyp_cube_fx17[row_col_fx17_target[0], row_col_fx17_target[1], :]

        # Record the rough and accurate pixel location of FX17 for checking.
        record_row_fx17_rough[row_fx10, col_fx10] = row_fx17_rough
        record_col_fx17_rough[row_fx10, col_fx10] = col_fx17_rough
        record_row_fx17_accu[row_fx10, col_fx10] = row_col_fx17_target[0]
        record_col_fx17_accu[row_fx10, col_fx10] = row_col_fx17_target[1]

    print('Row %d (%.2f percent) finished.' % (row_fx10, (row_fx10 / hyp_cube_fx10.shape[0]) * 100))

fused_hypcube = np.concatenate((hyp_cube_fx10, hyp_cube_fx17_to_fx10), axis=2)
fused_wave = np.concatenate((wave_fx10, wave_fx17))

print('Processing finished')
stop = datetime.now()
print('Total time: ', stop - start)


########################################################################################################################
# Show the results
########################################################################################################################
f1, ax_f1 = plt.subplots(1, 4)
f1.suptitle('Results of fusion', fontsize=12, fontweight='bold')
ax_f1[0].imshow(hyp_cube_fx17[:, :, 0], cmap='jet')
ax_f1[0].set_title('FX17 at ' + str(wave_fx17[0]) + ' nm')
ax_f1[1].imshow(hyp_cube_fx10[:, :, -1], cmap='jet')
ax_f1[1].set_title('FX10 at ' + str(wave_fx10[-1]) + ' nm')
ax_f1[2].imshow(fused_hypcube[:, :, 222], cmap='jet')
ax_f1[2].set_title('FX17 to FX10 at ' + str(fused_wave[222]))
ax_f1[3].imshow(xyz_fx10[:, :, 2], cmap='jet')
ax_f1[3].set_title('z of FX10')

f2, ax_f2 = plt.subplots(1, 2)
ax_f2[0].imshow(fused_hypcube[:, :, 222], cmap='jet')
ax_f2[0].set_title('Fused data at ' + str(fused_wave[222]))
ax_f2[0].plot(check_col_1, check_row_1, 'r+')
ax_f2[1].plot(fused_wave, fused_hypcube[check_row_1, check_col_1, :], color='red')
ax_f2[0].plot(check_col_2, check_row_2, 'y+')
ax_f2[1].plot(fused_wave, fused_hypcube[check_row_2, check_col_2, :], color='yellow')
ax_f2[1].set_title('Reflectance')
ax_f2[1].set_xlabel('Wavelength (nm)')
ax_f2[1].set_ylabel('Reflectance')

f2, ax_f2 = plt.subplots(1, 1)
diff = compare_images(np.asarray(hyp_cube_fx10)[:, :, -1], fused_hypcube[:, :, 222], method='diff')
ax_f2.set_title('Compare the difference of the original and fused image')
ax_f2.imshow(diff, cmap='jet')
plt.show()


########################################################################################################################
# Save the result
########################################################################################################################
if flag_save:
    dict = {'fused_hypcube': fused_hypcube,
            'fused_wave': fused_wave,
            'xyz': xyz_fx10,
            'search_radius': search_radius,
            'row_fx17_rough': record_row_fx17_rough,
            'col_fx17_rough': record_col_fx17_rough,
            'row_fx17_accurate': record_row_fx17_accu,
            'col_fx17_accurate': record_col_fx17_accu}

    joblib.dump(dict, 'fuse_fx17_to_fx10_' + data_name_fx10[-16:-4] + '.sav')

print('Data saved.')



