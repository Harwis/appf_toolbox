import sys
sys.path.append('E:/python_projects/appf_toolbox_project')
from appf_toolbox.hyper_processing import envi_funs
# from appf_toolbox.hyper_processing import pre_processing as pp
from matplotlib import pyplot as plt
import numpy as np
# from datetime import datetime


########################################################################################################################
# Parameters
########################################################################################################################
data_path = 'E:/Data/chikpea_Ali_Montana_0644'
data_name = 'swir_110_151_10010_2022-11-06_22-13-21'


band_num_check = 10
pix_check = [00, 150] # (col, row)
flag_save = True


########################################################################################################################
# Read data
########################################################################################################################
print('Reading data......')
raw_data, meta_plant = envi_funs.read_hyper_data(data_path, data_name)
ncols = meta_plant.ncols
nrows = meta_plant.nrows
nbands = meta_plant.nbands
wavelengths = np.zeros((meta_plant.metadata['Wavelength'].__len__(), ))
for i in range(wavelengths.size):
    wavelengths[i] = float(meta_plant.metadata['Wavelength'][i])


########################################################################################################################
# Calibrate the data
########################################################################################################################
hypcube = envi_funs.calibrate_hyper_data(raw_data['white'], raw_data['dark'], raw_data['plant'],
                                         trim_rate_t_w=0.1, trim_rate_b_w=0.95)

# Check an image
fig1 = plt.figure()
ax1_f1 = fig1.add_subplot(2,2,1)
ax1_f1.imshow(hypcube[:, :, band_num_check])
ax1_f1.scatter(pix_check[0], pix_check[1], color='red', marker='+')
ax1_f1.scatter(pix_check[0], pix_check[1], color='red', marker='o')
ax1_f1.set_title('Band ' + str(band_num_check) + ' ' + str(wavelengths[band_num_check]) + ' nm (Before smoothing)')

# Check a ref
a_ref = hypcube[pix_check[1], pix_check[0], :]
ax2_f1 = fig1.add_subplot(2, 2, 2)
ax2_f1.plot(wavelengths, a_ref, 'g', label='Not smoothed')



########################################################################################################################
# Save
########################################################################################################################
if flag_save:
    plt.imsave(data_name + '_man.png', hypcube[:, :, band_num_check])
    print(data_name + '_man.png was saved!')