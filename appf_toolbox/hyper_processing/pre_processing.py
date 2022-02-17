from spectral import *
import numpy as np
from matplotlib import pyplot as plt

########################################################################################################################
# remove_jumps
########################################################################################################################
def remove_jumps(spec_sig, ind_jumps=[], num_points=5, flag_auto_detect_jumps=False, t_der=0.004, r_win=15,
                 flag_plot_result=False):
    """
    Remove the jumps in a spectral signature.

    :param spec_sig: the spectral signature need to be fixed; 1D ndarray format

    :param ind_jumps: the indices of jumps; must be a list; the indices starts from 0; default is []; [650, 1480] for
           FieldSpec data; [398] for hyp images of vnir + swir

    :param  num_points: the number points to fit a line on the left-side of the jumps

    :param flag_auto_detect_jumps: the flag to determine if use auto detection of the jumps or not; default is False

    :param t_der: the threshold of the second derivative; if a value > t_der, then it is a jump; only used when
           flag_auto_detect-jumps == True; default is 0.004 for FieldSpec data; recommend 0.01 for imaging data.

    :param r_win: the radius of the window to distinguish jumps and randoms noises;
           only used when flag_auto_detect_jumps == True; default is 15.

    :param flag_plot_result: the flag to indicate if show the results in a figure or not; default is False

    :return: If ind_jumps is not empty, return the fixed spectral signature without jumps; otherwise return -1

    """

    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.optimize import curve_fit

    # Show a data point to check
    if flag_plot_result:
        fig = plt.figure()
        ax_raw = fig.add_subplot(3, 1, 1)
        ax_raw.plot(spec_sig, label='Original data')
        ax_raw.set_ylabel('Raw data', fontsize=12, fontweight='bold')

    ####################################################################################################################
    # Using the second-order derivative to automatically find the the jumps
    # der of f = (f(x + delt x) - f(x)) / delt x
    # Der of ref(0) is retained and the ref(n) is discarded
    ####################################################################################################################
    if flag_auto_detect_jumps:
        # First order derivative
        fir_der = np.delete(spec_sig, 0) - np.delete(spec_sig, spec_sig.shape[0] - 1)

        if flag_plot_result:
            ax_1der = fig.add_subplot(3, 1, 2)
            ax_1der.plot(fir_der)
            ax_1der.set_xlabel('band number', fontsize=12, fontweight='bold')
            ax_1der.set_ylabel('1st der', fontsize=12, fontweight='bold')
            # ax_1der.set_title('First order derivative')

        # Second order derivative
        sec_der = np.delete(fir_der, 0) - np.delete(fir_der, fir_der.shape[0] - 1)

        if flag_plot_result:
            ax_2der = fig.add_subplot(3, 1, 3)
            ax_2der.plot(sec_der)
            ax_2der.set_xlabel('band number', fontsize=12, fontweight='bold')
            ax_2der.set_ylabel('2end der', fontsize=12, fontweight='bold')
            ax_2der.hlines(t_der, 0, sec_der.shape[0] - 1, colors='red')
            ax_2der.text(0, t_der, 'Threshold')
            # ax_2der.set_title('second order derivative')

        # In values of the second order derivative, tow criterias are used to determine if there is a jump at the index
        # i. 1; t_der # 2: The number of high-values in the window of [i - r_win, i + r_win + 1] at i. For a real jump,
        # num_high = 2
        # First, zero-padding the borders.
        sec_der_pad = np.concatenate((np.zeros((r_win,)), sec_der, np.zeros((r_win,))))

        # Find the index of the jumps
        ind_jumps = []
        for i in range(sec_der.shape[0]):
            num_high_in_win = np.sum(np.abs(sec_der_pad[i:i + 2 * r_win + 1]) > t_der)
            if np.logical_and(np.abs(sec_der[i]) > t_der, num_high_in_win == 2):
                ind_jumps.append(i)

        if ind_jumps != []:
            # The high-values of 2end deri appears as paris. In each pair, remove the first one since only the second
            # one indicate the location of the jump.
            ind_jumps = np.delete(ind_jumps, range(0, ind_jumps.__len__(), 2))
    else:  # Use the manually inputted band numbers.
        pass

    ####################################################################################################################
    # Fix the jumps
    ####################################################################################################################
    def fit_a_line(x, a, b):
        return a * x + b

    if ind_jumps == []:
        print('No jumps found!')
        return -1
    else:
        # Fix the jump
        est_values = []
        for i_jump in ind_jumps:
            # Fit a line
            x1 = i_jump - num_points
            x2 = i_jump + 1

            # If fit to a horizontal line, curve_fit() works not well so have to manually set a and b.
            if np.all(spec_sig[x1:x2] == spec_sig[x1:x2][0]):
                a = 0
                b = spec_sig[x1:x2][0]
            else:
                popt, pcov = curve_fit(fit_a_line, np.arange(x1, x2), spec_sig[x1:x2])
                a = popt[0]
                b = popt[1]

            # Using the line to estimate y2. (x1, y1) is the first point for fitting the line. (x2, y2) is the first
            # point after the jump.
            y2 = fit_a_line(i_jump + 1, a, b)
            est_values.append(y2)

            if flag_plot_result:
                y1 = fit_a_line(x1, a, b)
                ax_raw.plot((x1, x2), (y1, y2), 'r--', linewidth=2)
                ax_raw.text(x2, y2 - 0.05, 'jump', fontsize=12)

        # Estimate the value after the jumps.
        ind_jumps = np.asarray(ind_jumps)
        est_values = np.asarray(est_values)
        raw_values = spec_sig[ind_jumps + 1]
        rel_shifts = raw_values - est_values
        abs_shifts = np.cumsum(rel_shifts)

        # Shift the data
        spec_sig_fixed = spec_sig.copy()
        for i in range(0, ind_jumps.shape[0]):
            if i < ind_jumps.shape[0] - 1:
                spec_sig_fixed[ind_jumps[i] + 1:ind_jumps[i + 1] + 1] = \
                    spec_sig[ind_jumps[i] + 1:ind_jumps[i + 1] + 1] - abs_shifts[i]
            else:
                spec_sig_fixed[ind_jumps[i] + 1:] = spec_sig[ind_jumps[i] + 1:] - abs_shifts[i]

        # Check if some of the values < 0 after shifting
        min_ref = np.min(spec_sig_fixed)
        if min_ref < 0:
            spec_sig_fixed = spec_sig_fixed - min_ref

        if flag_plot_result:
            ax_raw.plot(spec_sig_fixed, 'g--', label='Jumps fixed')
            ax_raw.legend()

        return spec_sig_fixed


########################################################################################################################
# Smooth_spectral_signature
########################################################################################################################
def smooth_savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0,
                           flag_fig = False, id_x=0):
    """
    Smooth curves using savgol filter. Call scipy.signal.savgol_filter.

    :param x: Can be 1D or 2D ndarray
    :param window_length:
    :param polyorder:
    :param deriv:
    :param delta:
    :param axis:
    :param mode:
    :param cval:
    :param flag_fig:
    :param id_x:
    :return:

    Author: Huajian Liu
    Email: huajian.liu@adelaide.edu.au

    Version: v0 (29, Nov, 2019)
    """

    # Call scipy.singal.savgol_filter
    from scipy import signal
    x_sm = signal.savgol_filter(x, window_length, polyorder, deriv, delta, axis, mode, cval)

    # Pick up a point to check
    if flag_fig:
        if x.shape.__len__() == 1:
            a_point = x
            a_point_sm = x_sm
        else:
            a_point = x[id_x]
            a_point_sm = x_sm[id_x]

        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(a_point, 'g', linewidth=3, label='Original')
        ax.plot(a_point_sm, 'r--', label='Smoothed')
        ax.set_title('Smooting using savgol filter')
        ax.legend()

    return x_sm


if __name__ == '__main__':
    print(smooth_savgol_filter_f.__doc__)

    from sklearn.externals import joblib

    data_path = 'demo_data'
    data_name = 'grass_vnir_n.sav'
    data_id = 10
    window_length = 11
    polyorder = 3
    id_data = 100

    ref_n = joblib.load(data_path + '/' + data_name)
    ref = ref_n['ref']
    data_sm = smooth_savgol_filter_f(ref, window_length, polyorder, flag_fig=True, id_x=id_data)


########################################################################################################################
# rotate_hypercube()
########################################################################################################################
def rotate_hypercube(hypercube, angle, scale=1, center='middle', flag_show_img=False, band_check=100):
    """
    Rotate a hypercube.

    :param hypercube: an ndarray of size (rows, cols, bands)
    :param angle: 0-360 degrees
    :param scale: the scale of images; in the range of (0, 1]; defual is 1.
    :param center: the center of the rotation; default is (cols/2, rows/2)
    :param flag_show_img: the flag to show the images or not
    :param band_check: The band at which the image will be checked.
    :return: the rotatd hypercube

    Author: Huajian Liu

    Version: v0 (Nov 25, 2019)
    """
    from matplotlib import pyplot as plt
    import cv2

    cols = hypercube.shape[1]
    rows = hypercube.shape[0]

    if center=='middle':
        # Use default values
        center = (cols/2, rows/2)
    else:
        # Use the (cols, rows) input by user.
        pass

    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # warp the image
    dst = cv2.warpAffine(hypercube, rotation_matrix, (cols, rows))

    if flag_show_img:
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(hypercube[:, :, band_check], cmap='gray')
        ax1.set_title('Original image')
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(dst[:, :, band_check], cmap='gray')
        ax2.set_title('Rotated image')

    return dst


########################################################################################################################
# Resample the spectral signatures according to the wavelength
########################################################################################################################
def spectral_resample(source_spectral_sig_array, source_wavelength, destination_wavelength, flag_fig=False, id_check=0):
    """
    Call spectral.BandResampler. Conduct spectral resampling.
    :param source_spectral_sig_array: can be 1D or 2D; each row is a spectral signature
    :param source_wavelength: 1D array
    :param destination_wavelength: 1D array; should in the range of [min, max] of source_wavelength
    :param flag_fig: flag to show the result or not
    :param id_check: the ID of the spectral signature to check
    :return: the destination spectral signature array which has the same shape of the source array.
    """

    # Make a resampler object
    resampler = BandResampler(source_wavelength, destination_wavelength)

    # Calculate the destination array.
    if source_spectral_sig_array.shape.__len__() == 1:
        destination_spectral_sig_array = resampler(source_spectral_sig_array)
    else:
        destination_spectral_sig_array = []
        for a_data in source_spectral_sig_array:
            destination_spectral_sig_array.append(resampler(a_data))
        destination_spectral_sig_array = np.asarray(destination_spectral_sig_array)

    # Show the result
    if flag_fig:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title('Source', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Wavelenght (nm)')
        ax1.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title('Destination', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Wavelenght (nm)')
        ax2.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Reflectance', fontsize=12, fontweight='bold')

        if source_spectral_sig_array.shape.__len__() == 1:
            source_spectral_sig_check = source_spectral_sig_array
            destination_spectral_sig_check = destination_spectral_sig_array
        else:
            source_spectral_sig_check = source_spectral_sig_array[id_check]
            destination_spectral_sig_check = destination_spectral_sig_array[id_check]

        ax1.plot(source_wavelength, source_spectral_sig_check)
        ax1.scatter(source_wavelength, source_spectral_sig_check, marker='o', s=5)
        ax2.plot(destination_wavelength, destination_spectral_sig_check)
        ax2. scatter(destination_wavelength, destination_spectral_sig_check, marker='o', s=5)

    return destination_spectral_sig_array


def green_plant_segmentation(data,
                             wavelength,
                             path_segmentation_model,
                             name_segmentation_model,
                             band_R=97,
                             band_G=52,
                             band_B=14,
                             gamma=0.8,
                             flag_remove_noise=True,
                             flag_check=False):
    """
    Conduct green plant segmentation using a pre-trained model.
    :param data: Calibrated hypercube in float ndarray format.
    :param wavelength: The corresponding wavelength of the data (hypercube).
    :param path_segmentation_model: The path of the pre-trained segmentation model.
    :param name_segmentation_model: The name of the pre-trained segmentation mode.
    :param band_R: An user-defined red band for checking the result. Defaul is 300.
    :param band_G: An user-defined green band for checking the result. Defaul is 200.
    :param band_B: An user-defined blue band for checking the result. Defaul is 100.
    :param gamma: gamma value for exposure adjustment.
    :param flag_remove_noise: Flag to remove the noise in BW image. Default is True.
    :param flag_check: Flag to show the results of segmenation.
    :return: The BW image and pseu image.

    Author: Huajina Liu
    Email: huajian.liu@adelaide.edu.au

    Date: Otc 13 2021
    Version: 0.0
    """

    import numpy as np
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent)
    from appf_toolbox.hyper_processing import pre_processing as pp
    from appf_toolbox.hyper_processing import transformation as tf
    from skimage import morphology
    from skimage import exposure
    import joblib

    ####################################################################################################################
    # Load segmentation model
    ####################################################################################################################
    model_record = joblib.load(path_segmentation_model + '/' + name_segmentation_model)
    model = model_record['model']

    ####################################################################################################################
    # Smooth
    ####################################################################################################################
    # Reshape
    row = data.shape[0]
    col = data.shape[1]

    # R, G and B bands for later use
    R = data[:, :, band_R].reshape((row, col, 1)).copy()
    G = data[:, :, band_G].reshape((row, col, 1)).copy()
    B = data[:, :, band_B].reshape((row, col, 1)).copy()

    data = data.reshape((row * col, data.shape[2]), order='C')

    # Smooth
    data = pp.smooth_savgol_filter(data, model_record['window length of smooth filter'],
                                   model_record['polyorder of smooth filter'])
    data[data < 0] = 0

    ####################################################################################################################
    # Classification
    ####################################################################################################################
    # Spectral resampling
    wave_model = model_record['wave_model']
    data = pp.spectral_resample(data, wavelength, wave_model)

    # hc2hhsi
    data = data.reshape((row, col, data.shape[1]))
    data, saturation, intensity = tf.hc2hhsi(data)

    # Reshape to 2D for OneClassSVM_swir_hh
    data = data.reshape((row * col, data.shape[2]), order='C')

    # Classification
    classes = model.predict(data)

    ####################################################################################################################
    # Make BW and pseu image
    ####################################################################################################################
    # BW image
    bw = classes.reshape((row, col), order='C')
    bw[bw == -1] = 0
    # bw[bw == 1] = 1

    # Remove noise in BW image
    if flag_remove_noise:
        # Erosion and reconstruction
        selem = np.ones((3, 3))
        bw_ero = morphology.binary_erosion(bw, selem=selem)
        bw = morphology.reconstruction(bw_ero, bw)

        # Remove_small holes; only accept bool type
        bw = bw.astype(bool)
        bw = morphology.remove_small_holes(bw)

    if np.sum(bw) == 0:
        print('No pixels of green plants was detected!')

    # pseu image
    border = np.logical_and(bw, np.bitwise_not(morphology.binary_erosion(bw)))

    R[border] = 1
    G[border] = 0
    B[border] = 0
    pseu = np.concatenate((R, G, B), axis=2)
    pseu = exposure.adjust_gamma(pseu, gamma)

    ####################################################################################################################
    # Check image
    ####################################################################################################################
    if flag_check:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(bw, cmap='gray')
        ax[0].set_title('BW image')
        ax[1].imshow(pseu)
        ax[1].set_title('Pseudo image R=' + str(wavelength[band_R]) + ' G=' + str(wavelength[band_G]) + ' B=' + str(wavelength[band_B]))
        fig.suptitle('Crop segmentation')
        plt.show()

    return bw, pseu



########################################################################################################################
# Green crop segmentation batch
########################################################################################################################
def green_plant_segmentation_batch(hyp_data_path,
                                   model_name_vnir='',
                                   model_name_swir='',
                                   model_path='./',
                                   gamma=0.8,
                                   flag_remove_noise=False,
                                   white_offset_top=0.1,
                                   white_offset_bottom=0.9,
                                   band_R=10,
                                   band_G=50,
                                   band_B=150,
                                   flag_save=True,
                                   save_path='./'):
    """
    Conduct green plant segmentaion for the WIWAM hyprspectral data uisng pre-trained models.
    :param hyp_data_path: Path of WIWAN hyperspectral data.
    :param model_name_vnir: Crop segmentation model for VNIR data
    :param model_name_swir: Crop segmentation model for SWIR data
    :param model_path: Path of the models
    :param gamma: For gamma correction of gray-scal images; [0.1, 1]
    :param flag_remove_noise: Flage for de-nose of BW image
    :param white_offset_top: Offset of the top of white reference; default  0.1
    :param white_offset_bottom: Offset of the bottom of white reference default 0.9
    :param band_R: Band number of R for making pseudo images
    :param band_G: Band number of G for making pseudo image
    :param band_B: Band number of B for making pseudo image
    :param flag_save: Flage for the saving the BW and pseudo image
    :param save_path: Path for saving the image.
    :return: 0

    Author: Huajina Liu
    Email: huajian.liu@adelaide.edu.au

    Date: Otc 13 2021
    Version: 0.0
    """

    # -------------------------------------------------------------
    # Import toolboxes
    # -------------------------------------------------------------
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent)
    from appf_toolbox.hyper_processing import transformation as tf
    from appf_toolbox.hyper_processing import pre_processing as pp
    from appf_toolbox.hyper_processing import envi_funs
    from skimage import morphology
    from skimage import exposure
    from matplotlib import pyplot as plt
    from os import walk

    # Read the data names for processing
    for (hyp_path, hyp_name, hyp_files) in walk(hyp_data_path):
        break

    # ------------------------------------------------------------
    # Process the data
    # ------------------------------------------------------------
    number_pro = 0
    error_report = []

    for i in range(0, hyp_name.__len__()):
        # ...............
        # Read a data set
        # ...............
        try:
            raw_data, meta_plant = envi_funs.read_hyper_data(hyp_data_path, hyp_name[i])
        except:
            error_ms = 'Read ENVI files errors in ' + hyp_name[i]
            print(error_ms)
            error_report.append(hyp_name[i])
            number_pro += 1
            continue

        # ...........
        # Calibration
        # ...........
        data = envi_funs.calibrate_hyper_data(raw_data['white'], raw_data['dark'], raw_data['plant'], white_offset_top,
                                              white_offset_bottom)

        # ...........
        # Segmentation
        # ...........
        wavelength = np.zeros((meta_plant.metadata['Wavelength'].__len__(),))
        for j in range(wavelength.size):
            wavelength[j] = float(meta_plant.metadata['Wavelength'][j])

        if hyp_name[i][0:4] == 'vnir':
            model_name = model_name_vnir
        else:
            model_name = model_name_swir

        bw, pseu = green_plant_segmentation(data,
                                            wavelength,
                                            path_segmentation_model=model_path,
                                            name_segmentation_model=model_name,
                                            band_R=band_R,
                                            band_G=band_G,
                                            band_B=band_B,
                                            gamma=gamma,
                                            flag_remove_noise=flag_remove_noise)

        # ............
        # Save results
        # ............
        if flag_save:
            plt.imsave(save_path + '/' + hyp_name[i] + '_bw.png', bw, cmap='gray')
            plt.imsave(save_path + '/' + hyp_name[i] + '_pseu.png', pseu)

        # ....................................
        # Print information for each iteration
        # ....................................
        number_pro += 1
        print(hyp_name[i] + ' finished!' + ' ' + str('%.2f' % (100 * number_pro / hyp_name.__len__())) + '%',
              '(' + str(number_pro) + '/' + str(hyp_name.__len__()) + ')' )

    print('Error report: ')
    print(error_report)

    return 0


def average_ref_under_mask(data,
                           mask,
                           flag_smooth=True,
                           window_length=21,
                           polyorder=3,
                           flag_check=False,
                           band_ind=20):
    """
    Calculate the average reflectance of the pixels of a hypercube under a mask.
    :param data: A calibrated hypercube in float ndarray format
    :param mask: A Bool image in which the pixels of object is True and background is False.
    :param flag_smooth: The flag to smooth the reflectance curve or not.
    :param window_length: The window length of SAVGOL filter.
    :param polyorder: The polyorder of SAVGOL filter.
    :param flag_check: The flag to check the result.
    :param band_ind: The band index for checking the results.
    :return: The average reflectance of the pixels under the mask.

    Author: Huajian Liu
    Email: huajian.liu@adelaide.edu.au
    Version: 0.0 date: Oct 14 2021
    """

    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent)
    from appf_toolbox.hyper_processing import pre_processing as pp
    from matplotlib import pyplot as plt

    # Check mask
    if not mask.any():
        print('The maks is all-zeros. ')
        return np.zeros((data.shape[2], ))
    else:
        # Take out pixels under the mask
        data[mask==0]=0

        # Check mask
        if flag_check:
            fig1, af1 = plt.subplots(3, 1)
            af1[0].imshow(data[:, :, band_ind], cmap='gray')
            af1[0].set_title('Image of band ' + str(band_ind))

        # Reshape and remove zero-rows
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]), order='C')
        zero_row_ind = np.where(~data.any(axis=1))[0]
        data = np.delete(data, zero_row_ind, axis=0)

        # Average
        ave_ref = np.mean(data, axis=0)

        # Check
        if flag_check:
          for ref_ind in range(0, data.shape[0], int(data.shape[0]/10)):
                af1[1].plot(data[ref_ind], linestyle='dashed')
          af1[1].set_ylabel('Reflectance', fontsize=12, fontweight='bold')
          af1[1].set_title('Reflectance before smoothing.')
          af1[1].plot(ave_ref, color='red', label='Average ref')
          plt.legend()

         # Smooth
        if flag_smooth:
            data = pp.smooth_savgol_filter(data, window_length, polyorder)
            data[data < 0] = 0

            # Average
            ave_ref = np.mean(data, axis=0)

            if flag_check:
                for ref_ind in range(0, data.shape[0], int(data.shape[0]/10)):
                    af1[2].plot(data[ref_ind], linestyle='dashed')
                af1[2].set_ylabel('Reflectance', fontsize=12, fontweight='bold')
                af1[2].set_title('Reflectance after smoothing.')
                af1[2].plot(ave_ref, color='red', label='Average ref')
                plt.legend()
                plt.show()

        return ave_ref


########################################################################################################################
# Calculate the average reflectance of crops for a wiwam hyp-image
########################################################################################################################
def ave_ref_under_mask_wiwam_batch(hyp_path,
                                   hyp_name,
                                   mask,
                                   flag_smooth=True,
                                   window_length=21,
                                   polyorder=3,
                                   flag_check=False,
                                   band_ind=50,
                                   ref_ind=50,
                                   white_offset_top=0.1,
                                   white_offset_bottom=0.9):
    """
    Calculate the average reflectance of crops using a mask (BW).
    If the mask is all-zeros, the average reflectance is set to np.zeros((1, wavelengths.shape[0])).
    :param hyp_path: The path of the hyperspectral data for processing.
    :param hyp_name: The name of the hyperspectral data for processing.
    :param mask: the BW image as a mask generated by crop_segmentation().
    :param flag_smooth: Flag for smooth the reflectance data or not; defaut is True
    :param window_length: window length for smoothing; default is 21
    :param polyorder: Polyorder for smoothing; default is 3
    :param flag_check: Flag for check the processed data or not; default is Fasle
    :param band_ind: A random band number for check; default is 50
    :param ref_ind: A random reflectance of plant for check; default is 50
    :param white_offset_top: Offset of the top of white reference; default  0.1
    :param white_offset_bottom: Offset of the bottom of white reference; default  0.9
    :return: Averaged reflectance value of crop and the corresponding wavelengths.

    Author: Huajian Liu
    v0: 1 March, 2021
    """

    # -------------------------------------------------------------
    # Import toolboxes
    # -------------------------------------------------------------
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent)
    from appf_toolbox.hyper_processing import pre_processing as pp
    from appf_toolbox.hyper_processing import envi_funs
    from matplotlib import pyplot as plt

    # -------------------------------------------------------------
    # Read and calibration
    # -------------------------------------------------------------
    try:
        raw_data, meta_plant = envi_funs.read_hyper_data(hyp_path, hyp_name)
    except:
        error_ms = 'Read ENVI files errors in ' + hyp_path + '/' + hyp_name
        print(error_ms)

    data = envi_funs.calibrate_hyper_data(raw_data['white'], raw_data['dark'], raw_data['plant'], white_offset_top,
                                          white_offset_bottom)

    wavelengths = meta_plant.metadata['Wavelength']
    wavelengths = np.asarray(wavelengths)
    wavelengths = wavelengths.astype(float)

    # -------------------------------------------------------------
    # Average
    # -------------------------------------------------------------
    ave_ref = verage_ref_under_mask(data,
                                    mask,
                                    flag_smooth=flag_smooth,
                                    window_length=window_length,
                                    polyorder=polyorder,
                                    flag_check=flag_check,
                                    band_ind=band_ind)

    return ave_ref, wavelengths


#
