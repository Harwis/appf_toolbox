# Caculate the average reflectance of plants using the masks.
# Use the barcode_hypname .xslx file.
# The ref of missing pots (no hypname) is assigned 'No hyper-data'
# If mask is all-zeros, ref = all_zeros

import numpy as np
import sys
sys.path.append('/media/huajian/Files/python_projects/appf_toolbox_project')
from appf_toolbox.hyper_processing.pre_processing import ave_ref
from appf_toolbox.hyper_processing.pre_processing import remove_jumps
from matplotlib import pyplot as plt
import pandas as pd

########################################################################################################################
# Parameters
########################################################################################################################
# Paramters
hyp_data_path = '/media/huajian/TOSHIBA EXT/crown_rot_0590_top_view'

barcode_hypname_path = '/media/huajian/Files/python_projects/crown_rot_0590/crown_rot_0590_processed_data/crown_rot_0590_top_view'
barcode_hypname_file = 'barcode_hypname.xlsx'
sheet_name = 'barcode_hypname'

mask_path = '/media/huajian/Files/python_projects/crown_rot_0590/crown_rot_0590_processed_data/crown_rot_0590_top_view/segmentation'

t_wave_1 = 450
t_wave_2 = 1000
t_wave_3 = 2400

flag_smooth = True
flag_check = False
flag_save = True


########################################################################################################################
# Processing
########################################################################################################################
# Read barcode_hypname
barcode_hypname = pd.read_excel(barcode_hypname_path + '/' +barcode_hypname_file, sheet_name=sheet_name)

ref_list = []
count = 0
for a_row in barcode_hypname.iterrows():
    if pd.isnull(a_row[1]['vnir']):
        ref_list.append('No hyper-data')
        print('No hyper-data.')
    else:
        mask_vnir = plt.imread(mask_path + '/' + a_row[1]['vnir'] + '_bw.png')
        mask_swir = plt.imread(mask_path + '/' + a_row[1]['swir'] + '_bw.png')
        mask_vnir = mask_vnir[:, :, 0]
        mask_swir = mask_swir[:, :, 0]
        ave_ref_vnir, wave_vnir = ave_ref(hyp_data_path,
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

        ave_ref_swir, wave_swir = ave_ref(hyp_data_path,
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

        # Trim noisy bands
        wave_mask_vnir = np.logical_and(np.logical_or(wave_vnir > t_wave_1, wave_vnir == t_wave_1),
                                        np.logical_or(wave_vnir < t_wave_2, wave_vnir == t_wave_2))
        wave_vnir = wave_vnir[wave_mask_vnir]
        ave_ref_vnir = ave_ref_vnir[wave_mask_vnir]

        wave_mask_swir = np.logical_and(wave_swir > t_wave_2,
                                        np.logical_or(wave_swir < t_wave_3, wave_swir == t_wave_3))
        wave_swir = wave_swir[wave_mask_swir]
        ave_ref_swir = ave_ref_swir[wave_mask_swir]

        # Cat
        wave = np.concatenate((wave_vnir, wave_swir))
        ref = np.concatenate((ave_ref_vnir, ave_ref_swir))

        # Remove jump
        if (not mask_vnir.any()) or (not mask_swir.any()):
            pass
        else:
            ref = remove_jumps(ref, ind_jumps=[wave_vnir.shape[0] - 1], flag_plot_result=flag_check)

        # Check
        if flag_check:
            fig1, af1 = plt.subplots(1, 1)
            af1.plot(wave, ref)
            af1.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
            af1.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
            plt.pause(3)

        # Save to list
        ref_list.append(ref)

    # Count the processing
    count += 1
    print(str(count) + ' have finished (' + str(np.round(100 * count / barcode_hypname.shape[0], 2)) + '%).')


########################################################################################################################
# Save to hard disk
########################################################################################################################
if flag_save:
    hypdata = pd.DataFrame(ref_list)
    hypdata.columns = wave
    barcode_hypdata = pd.concat([barcode_hypname, hypdata], axis=1)

    writer = pd.ExcelWriter(barcode_hypname_path + '/' + 'barcode_hypdata.xlsx', engine='xlsxwriter')
    barcode_hypdata.to_excel(writer, sheet_name='barcode_hypdata')
    writer.save()
    print('Data saved!')


