# This demo shows how to read and calibrate the hyperspectral data in envi format.
import sys
sys.path.append('E:/python_projects/appf_toolbox_project')
from appf_toolbox.hyper_processing import envi_funs
from matplotlib import pyplot as plt
import numpy as np

########################################################################################################################
# Parameters
folder_path = 'C:/Users/Huajian/OneDrive - University of Adelaide/appf_toolbox_demo_data'
folder_name = 'vnir_74_104_6235_2019-10-18_01-47-33'


# Will read the wavelength on every band_interval
band_interval = 10

# A random band number for checking
band_num = 200

# Trim the top and bottom of white reference for calibration
trim_white_top = 0.0001
trim_white_bot = 0.9999
########################################################################################################################


# Read data
raw_data, meta_target = envi_funs.read_hyper_data(folder_path, folder_name)
print('The size of white ref: ', raw_data['white'].shape)
print('The size of dark ref: ', raw_data['dark'].shape)
print('The size of object: ', raw_data['plant'].shape)
n_samples = meta_target.ncols
n_bands = meta_target.nbands

# The corresponding wavelength
wavelengths = meta_target.metadata['Wavelength']
wavelengths = np.asarray(wavelengths)
wavelengths = wavelengths.astype(float)


# Calibration (Calculate the reflectance data of plants).
# Average the white and dark references over the lines;
white_raw = raw_data['white']
white_raw = white_raw[int(white_raw.shape[0] * trim_white_top):int(white_raw.shape[0] * trim_white_bot), :, :]
ave_white = np.mean(white_raw, 0)
ave_white = ave_white.reshape((1, n_samples, n_bands))
ave_while = np.tile(ave_white, (raw_data['plant'].shape[0], 1, 1))
ave_dark = np.mean(raw_data['dark'], 0)
ave_dark = ave_dark.reshape((1, n_samples, n_bands))
ave_dark = np.tile(ave_dark, (raw_data['plant'].shape[0], 1, 1))
plant_cal = (raw_data['plant'] - ave_dark) / (ave_white - ave_dark + 1e-6)


########################################################################################################################
# Calculate the signal to noise ratio in total and each band; plot
n_noise = np.sum(np.logical_or(plant_cal > 1, plant_cal < 0))
sn_tot = n_noise / (n_bands * n_samples * plant_cal.shape[0])
print('Total noise (%) is ' + '%.2f' % (sn_tot * 100) + '%')

sn_band_list = []
for ind_band in range(n_bands):
    n_noise = np.sum(np.logical_or(plant_cal[:, :, ind_band] > 1, plant_cal[:, :, ind_band] < 0))
    sn_band = n_noise / (n_samples * plant_cal.shape[0])
    sn_band_list.append(sn_band)

fig_sn = plt.figure()
fig_sn.suptitle('Noise (%) at each band', fontsize=14, fontweight='bold')
plt.plot(wavelengths, np.asarray(sn_band_list) * 100)
plt.xlabel('Wavelengths (nm)', fontsize=12, fontweight='bold')
plt.ylabel('Noise (%)', fontsize=12, fontweight='bold')
########################################################################################################################


########################################################################################################################
# Plot the images one by one
fig_img = plt.figure()
ax1 = fig_img.add_subplot(2,2,1)
ax1.title.set_text('White ref')
ax2 = fig_img.add_subplot(2,2,2)
ax2.title.set_text('Dark ref')
ax3 = fig_img.add_subplot(2,2,3)
ax3.title.set_text('Raw image')
ax4 = fig_img.add_subplot(2,2,4)
for ind_band in range(0, n_bands + 1, band_interval):
    fig_img.suptitle(folder_name + '\n' + str(wavelengths[ind_band]) + ' nm @ band ' + str(ind_band), fontsize=14,
                     fontweight='bold')

    # Plant images @ band_num
    # Take out a 2D image @ band_num
    dark_img = raw_data['dark'][:, :, ind_band]
    white_img = white_raw[:, :, ind_band]
    plant_img_raw = raw_data['plant'][:, :, ind_band]

    # Reshape them to 2D images
    dark_img = dark_img.reshape(dark_img.shape[0], dark_img.shape[1])
    white_img = white_img.reshape(white_img.shape[0], white_img.shape[1])
    plant_img_raw = plant_img_raw.reshape(plant_img_raw.shape[0], plant_img_raw.shape[1])

    # The calibrated image
    plant_img_cal = plant_cal[:, :, ind_band]

    # Show the raw images
    ax1.imshow(white_img, cmap='jet')
    ax2.imshow(dark_img, cmap='jet')
    ax3.imshow(plant_img_raw, cmap='jet')

    ax4.clear()
    ax4.title.set_text('Calibrated image (yellow noise >1, red noise < 0)')
    ax4.imshow(plant_img_cal, cmap='gray')

    ind_noise_type1 = np.argwhere(plant_img_cal > 1)
    ind_noise_type2 = np.argwhere(plant_img_cal < 0)

    if ind_noise_type1.shape[0] != 0:
        ax4.plot(ind_noise_type1[:, 1], ind_noise_type1[:, 0], 'r+')

    if ind_noise_type2.shape[0] != 0:
        ax4.plot(ind_noise_type2[:, 1], ind_noise_type2[:, 0], 'y+')

    if n_bands - ind_band - 1 < band_interval:
        plt.show()
    else:
        plt.pause(1)
#######################################################################################################################







