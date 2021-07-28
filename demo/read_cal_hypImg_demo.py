from appf_toolbox.hyper_processing import envi_funs
import numpy as np
from matplotlib import pyplot as plt


# Parameters
data_path = '/media/huajian/Files/python_projects/appf_toolbox_demo_data'
data_name = 'vnir_74_104_6235_2019-10-18_01-47-33'
band_num_check = 100
pix_check = [130, 262]

# Read the data
raw_data, meta_plant = envi_funs.read_hyper_data(data_path, data_name)
wavelengths = np.zeros((meta_plant.metadata['Wavelength'].__len__(), ))
for i in range(wavelengths.size):
    wavelengths[i] = float(meta_plant.metadata['Wavelength'][i])

# Calibrate the data
hypcube = envi_funs.calibrate_hyper_data(raw_data['white'], raw_data['dark'], raw_data['plant'],
                                         trim_rate_t_w=0.1, trim_rate_b_w=0.95)

# Plot to check
fig1 = plt.figure()
ax1_f1 = fig1.add_subplot(1,2,1)
ax1_f1.imshow(hypcube[:, :, band_num_check], cmap='gray')
ax1_f1.scatter(pix_check[1], pix_check[0], color='red', marker='+')
ax1_f1.set_title('Calibrated image at band ' + str(band_num_check) + '@' + str(wavelengths[band_num_check]) + 'nm')
ax2_f1 = fig1.add_subplot(1,2,2)
ax2_f1.plot(wavelengths, hypcube[pix_check[0], pix_check[1], :])
ax2_f1.set_xlabel('Wavelengths (nm)', fontsize=12, fontweight='bold')
ax2_f1.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
ax2_f1.set_title('Reflectance at row ' + str(pix_check[0]) + ' col ' + str(pix_check[1]))