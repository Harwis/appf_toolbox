"""
my_envi_funs module defines functions to process hyperspectral data in envi format. 
"""

from spectral import *
import numpy as np

def read_hyper_data(path, folder_name):
    """
    Read hyperspectral data in ENVI fromat.

    :param path: the path containing the folders of the data.
    :param folder_name: the name of the data folder.
    :return: {'white:', hypercube_of_white, 'dark:', hypercube_of_dark, 'plant:', hypercube_of_object}, meta_of_plant\

    Author: Huajian liu
    version: v0 (10 May, 2018)
    """
    import spectral.io.envi as envi
    spectral.settings.envi_support_nonlowercase_params = True

    # Reading data
    meta_white = envi.open(path + '/' + folder_name + '/' + 'capture' + '/' + 'WHITEREF_' + folder_name + '.hdr',
                           path + '/' + folder_name + '/' + 'capture' + '/' + 'WHITEREF_' + folder_name + '.raw')

    meta_dark = envi.open(path + '/' + folder_name + '/' + 'capture' + '/' + 'DARKREF_' + folder_name + '.hdr',
                          path + '/' + folder_name + '/' + 'capture' + '/' + 'DARKREF_' + folder_name + '.raw')

    meta_plant = envi.open(path + '/' + folder_name + '/' + 'capture' + '/' + folder_name + '.hdr',
                           path + '/' + folder_name + '/' + 'capture' + '/' + folder_name + '.raw')

    return {'white': meta_white.load(), 'dark': meta_dark.load(), 'plant': meta_plant.load()}, meta_plant


def calibrate_hyper_data(white, dark, plant, trim_rate_t_w=0.05, trim_rate_b_w=0.95):
    """
    Calibrate hyerpsectral data.

    :param white: the hypercube of white returned from read_hyper_data()
    :param dark: the hypercube of dark returned from read_hyper_data()
    :param object: the hypercube of object returned from read_hyper_data()
    :param trim_rate_t_w: the rate for trimming the top of white
    :param trim_rate_b_w: the rate for trmming the bottom of white
    :return: the calibrated hypercube of object in [0 1]

    Author: Huajian liu
    version: v0 (10 May, 2018)
    """
    lines_w = white.shape[0]
    lines_o = plant.shape[0]
    samples = white.shape[1]
    bands = white.shape[2]

    # Take of the ROI of white
    white = white[int(lines_w * trim_rate_t_w):int(lines_w * trim_rate_b_w), 0:samples, 0:bands]

    # Make mean-images of white and dark. The number of lines of mean-images is the same as that of plant.
    white_mean = white.mean(0)
    white_mean = white_mean.reshape(1, samples, bands)
    white_mean = np.tile(white_mean, [lines_o, 1, 1])

    dark_mean = dark.mean(0)
    dark_mean = dark_mean.reshape(1, samples, bands)
    dark_mean = np.tile(dark_mean, [lines_o, 1, 1])

    plant_cal = (plant - dark_mean) / (white_mean - dark_mean + 1e-10)
    plant_cal[plant_cal > 1] = 1
    plant_cal[plant_cal < 0] = 0
    
    return plant_cal


######################################
# ENVI hyp-img pre-processing pipeline
######################################
def envi_hyp_img_pre_processing_pipeline(data_path, data_name, wavelength_low, wavelength_high,
                                 pix_check, band_num_check, window_length, polyorder, angle_rotation=0, flag_fig=False):
    """
    Conduct a pre-processing pipeline for envi hyperspectral images, including 1.reading an envi image, 2.calibrating the
    image, 3. rotate the image if necessary, smoothing and removing noisy bands.
    :param data_path: The path of the envi image.
    :param data_name: The name of the envi image.
    :param wavelength_low: The low-end of noisy wavelength.
    :param wavelength_high: The high-end of noisy wavelength.
    :param pix_check: [row, col] of pixel for checking.
    :param band_num_check: The band number for checking.
    :param window_length: The window length for smooth_savgol_filter.
    :param polyorder: The polyorder for smooth_savgol_filter.
    :param angle_rotation: The angle of image rotation. Default is 0.
    :param flag_fig: The flag to show the figure or not. Default is False.
    :return: The processed hypcube and the correspond wavelengths.
    """
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent)
    from appf_toolbox.hyper_processing import envi_funs
    from appf_toolbox.hyper_processing import pre_processing as pp
    from matplotlib import pyplot as plt
    import numpy as np

    # ---------
    # Read data
    # ---------
    print('Reading data......')
    raw_data, meta_plant = envi_funs.read_hyper_data(data_path, data_name)
    ncols = meta_plant.ncols
    nrows = meta_plant.nrows
    nbands = meta_plant.nbands
    wavelengths = np.zeros((meta_plant.metadata['Wavelength'].__len__(),))
    for i in range(wavelengths.size):
        wavelengths[i] = float(meta_plant.metadata['Wavelength'][i])

    # ------------------
    # Calibrate the data
    # ------------------
    hypcube = envi_funs.calibrate_hyper_data(raw_data['white'], raw_data['dark'], raw_data['plant'],
                                             trim_rate_t_w=0.1, trim_rate_b_w=0.95)
    # --------
    # Rotation
    # --------
    hypcube = pp.rotate_hypercube(hypcube, angle_rotation, band_check=100)

    # --------------
    # Check an image
    # --------------
    if flag_fig:
        fig1 = plt.figure()
        ax1_f1 = fig1.add_subplot(2, 2, 1)
        ax1_f1.imshow(hypcube[:, :, band_num_check])
        ax1_f1.scatter(pix_check[0], pix_check[1], color='red', marker='+')
        ax1_f1.scatter(pix_check[0], pix_check[1], color='red', marker='o')
        ax1_f1.set_title(
            'Band ' + str(band_num_check) + ' ' + str(wavelengths[band_num_check]) + ' nm (Before smoothing)')

        # Check a ref
        a_ref = hypcube[pix_check[1], pix_check[0], :]
        ax2_f1 = fig1.add_subplot(2, 2, 2)
        ax2_f1.plot(wavelengths, a_ref, 'g', label='Not smoothed')
        ax2_f1.set_title('Smoothing')

    # ------
    # Smooth
    # ------
    hyp_flat = hypcube.reshape((nrows * ncols, nbands), order='C')
    hyp_flat = pp.smooth_savgol_filter(hyp_flat, window_length=window_length, polyorder=polyorder)
    hyp_flat[hyp_flat < 0] = 0
    hyp_flat[hyp_flat > 1] = 1
    hypcube = hyp_flat.reshape((nrows, ncols, nbands), order='C')
    a_ref = hypcube[pix_check[1], pix_check[0], :]

    # Check after smoothing
    if flag_fig:
        ax2_f1.plot(wavelengths, a_ref, 'r--', label='Smoothed')
        ax3_f1 = fig1.add_subplot(2, 2, 3)
        ax3_f1.imshow(hypcube[:, :, band_num_check])
        ax3_f1.scatter(pix_check[0], pix_check[1], color='red', marker='+')
        ax3_f1.set_title(
            'Band ' + str(band_num_check) + ' ' + str(wavelengths[band_num_check]) + ' nm (After smoothing)')
        ax2_f1.legend()

    # -----------------
    # Remove noise band
    # -----------------
    good_band_ind = np.logical_and(
        np.logical_or(wavelengths > wavelength_low, wavelengths == wavelength_low),
        np.logical_or(wavelengths < wavelength_high, wavelengths == wavelength_high))
    good_band_ind = np.where(good_band_ind)[0]

    hypcube = hypcube[:, :, good_band_ind]
    wavelengths = wavelengths[good_band_ind]
    a_ref = hypcube[pix_check[1], pix_check[0], :]

    # Check after removing the noisy bands
    if flag_fig:
        ax4_f1 = fig1.add_subplot(2, 2, 4)
        ax4_f1.plot(wavelengths, a_ref, 'r', linewidth=2, label='Smoothed and removed noisy bands')
        ax4_f1.set_xlabel('Wavelengths (nm)', fontsize=12, fontweight='bold')
        ax4_f1.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
        ax4_f1.set_title('Smoothed and removed noisy bands')

    return hypcube, wavelengths
