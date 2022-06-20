# Caculate the average reflectance of plants using the masks.
# Use the barcode_hypname .xslx file.
# The ref of missing pots (no hypname) is assigned 'No hyper-data'
# If mask is all-zeros, ref = all_zeros

import numpy as np
import sys
# sys.path.append('/media/huajian/Files/python_projects/appf_toolbox_project')
sys.path.append('E:/python_projects/appf_toolbox_project')
from appf_toolbox.hyper_processing.pre_processing import statistics_under_mask_envi_file
from appf_toolbox.hyper_processing.pre_processing import remove_jumps
from appf_toolbox.hyper_processing import transformation as tr
from matplotlib import pyplot as plt
import pandas as pd
from sklearn .decomposition import PCA

########################################################################################################################
# Parameters
########################################################################################################################
# Paramters
hyp_data_path = 'E:/Data/ruby_0614'

barcode_hypdata_path = 'E:/python_projects/ruby_0614'
barcode_hypdata_file = 'barcode_hyperspectral_data.xlsx'
sheet_name = 'barcode_treatment_hypname'

mask_path = 'E:/Data/ruby_0614_segmentation_v2'

flag_trim_noisy_band = True
t_wave_1 = 450
t_wave_2 = 1000
t_wave_3 = 2400

n_comp_pca = 20

flag_smooth = True
flag_check = False
flag_save = True


########################################################################################################################
# Processing
########################################################################################################################
# Read barcode and hypname
barcode_treat_hypname = pd.read_excel(barcode_hypdata_path + '/' + barcode_hypdata_file, sheet_name=sheet_name)

ref_list = []
std_list = []
n_pixel_list = []
count = 0
for a_row in barcode_treat_hypname.iterrows():
    if pd.isnull(a_row[1]['vnir']):
        ref_list.append('No hyper-data')
        print('No hyper-data.')
    else:
        try:
            mask_vnir = plt.imread(mask_path + '/' + a_row[1]['vnir'] + '_bw.png')
        except FileNotFoundError as error:
            # mask_vnir = np.zeros((2, 2, 3)) # Cheat the program
            print('Error: ', error)
        mask_vnir = mask_vnir[:, :, 0]

        try:
            mask_swir = plt.imread(mask_path + '/' + a_row[1]['swir'] + '_bw.png')
        except FileNotFoundError as error:
            # mask_swir = np.zeros((2, 2, 3))

            print('Error: ', error)
        mask_swir = mask_swir[:, :, 0]

        stat_vnir = statistics_under_mask_envi_file(hyp_data_path,
                                                       a_row[1]['vnir'],
                                                       mask_vnir,
                                                       flag_smooth=flag_smooth,
                                                       # window_length=21,
                                                       # polyorder=3,
                                                       flag_check=flag_check,
                                                       # band_ind=50,
                                                       # ref_ind=50,
                                                       # white_offset_top=0.1,
                                                       # white_offset_bottom=0.9
                                                       )

        stat_swir = statistics_under_mask_envi_file(hyp_data_path,
                                                       a_row[1]['swir'],
                                                       mask_swir,
                                                       flag_smooth=flag_smooth,
                                                       # window_length=21,
                                                       # polyorder=3,
                                                       flag_check=flag_check,
                                                       # band_ind=50,
                                                       # ref_ind=50,
                                                       # white_offset_top=0.1,
                                                       # white_offset_bottom=0.9
                                                       )

        wave_vnir = stat_vnir['wavelength']
        wave_swir = stat_swir['wavelength']

        if flag_trim_noisy_band:
            # Trim noisy bands of vnir
            wave_mask_vnir = np.logical_and(np.logical_or(wave_vnir > t_wave_1, wave_vnir == t_wave_1),
                                            np.logical_or(wave_vnir < t_wave_2, wave_vnir == t_wave_2))
            wave_vnir = wave_vnir[wave_mask_vnir]
            ave_ref_vnir = stat_vnir['ave_ref'][wave_mask_vnir]
            std_ref_vnir = stat_vnir['std_ref'][wave_mask_vnir]

            # Trim noisy bands of swir
            wave_mask_swir = np.logical_and(wave_swir > t_wave_2,
                                            np.logical_or(wave_swir < t_wave_3, wave_swir == t_wave_3))
            wave_swir = wave_swir[wave_mask_swir]
            ave_ref_swir = stat_swir['ave_ref'][wave_mask_swir]
            std_ref_swir = stat_swir['std_ref'][wave_mask_swir]
        else:
            ave_ref_vnir = stat_vnir['ave_ref']
            std_ref_vnir = stat_vnir['std_ref']
            ave_ref_swir = stat_swir['ave_ref']
            std_ref_swir = stat_swir['std_ref']

        # Cat
        wave = np.concatenate((wave_vnir, wave_swir))
        ave_ref = np.concatenate((ave_ref_vnir, ave_ref_swir))
        std_ref = np.concatenate((std_ref_vnir, std_ref_swir))

        # Remove jump
        if (not mask_vnir.any()) or (not mask_swir.any()):
            pass
        else:
            ave_ref = remove_jumps(ave_ref, ind_jumps=[wave_vnir.shape[0] - 1], flag_plot_result=flag_check)
            std_ref = remove_jumps(std_ref, ind_jumps=[wave_vnir.shape[0] - 1], flag_plot_result=flag_check)

        # Check
        if flag_check:
            fig1, af1 = plt.subplots(1, 1)
            af1.plot(wave, ave_ref, color='red', linestyle='-', label='Average reflectance')
            af1.plot(wave, std_ref, color='red', linestyle='-.', label='STD')
            af1.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
            af1.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
            plt.legend()
            plt.pause(3)

        # Some other statistics
        n_pixel_vnir = stat_vnir['n_pixel']
        n_pixel_swir = stat_swir['n_pixel']
        n_pixel = [n_pixel_vnir, n_pixel_swir]

        # Save to list
        ref_list.append(ave_ref)
        std_list.append(std_ref)
        n_pixel_list.append(n_pixel)

    # Count the processing
    count += 1
    print(str(count) + ' have finished (' + str(np.round(100 * count / barcode_treat_hypname.shape[0], 2)) + '%).')


########################################################################################################################
# Save to hard disk
########################################################################################################################
if flag_save:
    import openpyxl as opx

    book = opx.load_workbook(filename=barcode_hypdata_path + '/' + barcode_hypdata_file)
    writer = pd.ExcelWriter(barcode_hypdata_path + '/' + barcode_hypdata_file, engine='openpyxl')
    writer.book = book

    # ave_ref
    df_ave_ref = pd.DataFrame(ref_list)
    df_ave_ref.columns = wave
    df_ave_ref = pd.concat([pd.DataFrame(barcode_treat_hypname, columns=['id_tag']), df_ave_ref], axis=1)
    df_ave_ref.to_excel(writer, sheet_name='Average reflectance')

    # STD
    df_std = pd.DataFrame(std_list)
    df_std.columns = wave
    df_std = pd.concat([pd.DataFrame(barcode_treat_hypname, columns=['id_tag']), df_std], axis=1)
    df_std.to_excel(writer, sheet_name='STD of reflectance')

    # Hyper-hue, saturation and intensity
    ref = np.asarray(ref_list)
    ref = ref.reshape((ref.shape[0], 1, ref.shape[1])) # Now, ref is a 3D ndarray
    hyper_hue, saturation, intensity = tr.hc2hhsi(ref)
    hyper_hue = hyper_hue.reshape((hyper_hue.shape[0], hyper_hue.shape[2]))
    hhsi = np.concatenate((hyper_hue,
                           saturation.reshape((saturation.shape[0], 1)),
                           intensity.reshape((intensity.shape[0], 1))), axis=1)
    df_hhsi = pd.DataFrame(hhsi)

    column_name = []
    for i in range(0, hyper_hue.shape[1]):
        column_name.append('hyper-hue_' + str(i))
    column_name.append('Saturation')
    column_name.append('Intensity')
    df_hhsi.columns = column_name

    df_hhsi = pd.concat([pd.DataFrame(barcode_treat_hypname, columns=['id_tag']), df_hhsi], axis=1)
    df_hhsi.to_excel(writer, sheet_name='Hyper-hue, sat and int')

    # SNV
    snv = tr.snv(np.asarray(ref_list))
    df_snv = pd.DataFrame(snv)
    df_snv.columns = wave
    df_snv = pd.concat([pd.DataFrame(barcode_treat_hypname, columns=['id_tag']), df_snv], axis=1)
    df_snv.to_excel(writer, sheet_name='snv')

    # First-order derivative
    fir_der = tr.first_order_derivative(np.asarray(ref_list))
    df_fir_der = pd.DataFrame(fir_der)
    df_fir_der.columns = wave[0:-1]
    df_fir_der = pd.concat([pd.DataFrame(barcode_treat_hypname, columns=['id_tag']), df_fir_der], axis=1)
    df_fir_der.to_excel(writer, sheet_name='1st-der')

    # Second-order derivative
    sec_der = tr.second_order_derivative(np.asarray(ref_list))
    df_sec_der = pd.DataFrame(sec_der)
    df_sec_der.columns = wave[0:-2]
    df_sec_der = pd.concat([pd.DataFrame(barcode_treat_hypname, columns=['id_tag']), df_sec_der], axis=1)
    df_sec_der.to_excel(writer, sheet_name='2end-der')

    # PCA
    pca = PCA(n_components=n_comp_pca)
    pcs = pca.fit_transform(np.asarray(ref_list))
    df_pcs = pd.DataFrame(pcs)

    column_name = []
    for i in range(0, pcs.shape[1]):
        column_name.append('pc' + str(i))
    df_pcs.columns = column_name

    df_pcs = pd.concat([pd.DataFrame(barcode_treat_hypname, columns=['id_tag']), df_pcs], axis=1)
    df_pcs.to_excel(writer, sheet_name='Principal component')

    # n_pixel
    df_n_pixel = pd.DataFrame(n_pixel_list)
    df_n_pixel.columns = ['n_pixel_vnir', 'n_pixel_swir']
    df_n_pixel = pd.concat([pd.DataFrame(barcode_treat_hypname, columns=['id_tag']), df_n_pixel], axis=1)
    df_n_pixel.to_excel(writer, sheet_name='Number of pixels under mask')

    writer.save()
    writer.close()
    print('Data saved!')


