import sys
sys.path.append('/media/huajian/Files/python_projects/appf_toolbox_project')
import numpy as np
from matplotlib import pyplot as plt
from appf_toolbox.hyper_processing import envi_funs
from appf_toolbox.hyper_processing import pre_processing as pp

########################################################################################################################
# Parameters
########################################################################################################################
data_path = '/media/huajian/Files/python_projects/appf_toolbox_demo_data'
data_name = 'vnir_74_104_6235_2019-10-18_01-47-33'
band_num_check = 100
pix_check = [255, 206] # (col, row)

# For smooth filter
wl = 21 # windows length
po = 3 # Polyorder


########################################################################################################################
# Read the data
########################################################################################################################
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


########################################################################################################################
# Rotate the hypercube 180 degree if necessary
########################################################################################################################
hypcube = pp.rotate_hypercube(hypcube, 180)

# Check an image
fig1 = plt.figure()
ax1_f1 = fig1.add_subplot(2,2,1)
ax1_f1.imshow(hypcube[:, :, band_num_check])
ax1_f1.scatter(pix_check[0], pix_check[1], color='red', marker='+')
ax1_f1.set_title('Band ' + str(band_num_check) + ' ' + str(wavelengths[band_num_check]) + ' nm (Before smoothing)')

# Check a ref
a_ref = hypcube[pix_check[0], pix_check[1], :]
ax2_f1 = fig1.add_subplot(2,2,2)
ax2_f1.plot(wavelengths, a_ref, 'g', label='Not smoothed')
ax2_f1.set_title('Smoothing')


########################################################################################################################
# Fix the jumps if necessary.
########################################################################################################################


########################################################################################################################
# Smooth
########################################################################################################################
hyp_flat = hypcube.reshape((nrows * ncols, nbands), order='C')
hyp_flat = pp.smooth_savgol_filter(hyp_flat, window_length=wl, polyorder=po)
hypcube = hyp_flat.reshape((nrows, ncols, nbands), order='C')
a_ref = hypcube[pix_check[0], pix_check[1], :]

# Check after smoothing
ax2_f1.plot(wavelengths, a_ref, 'r--', label='Smoothed')
ax3_f1 = fig1.add_subplot(2,2,3)
ax3_f1.imshow(hypcube[:, :, band_num_check])
ax3_f1.scatter(pix_check[0], pix_check[1], color='red', marker='+')
ax3_f1.set_title('Band ' + str(band_num_check) + ' ' + str(wavelengths[band_num_check]) + ' nm (After smoothing)' )


########################################################################################################################
# Remove noise band
########################################################################################################################
data_type = data_name[0:4]
if data_type== 'vnir':
    good_band_start = 42
    good_band_stop = 440
else:
    good_band_start = 0
    good_band_stop = 251

hypcube = hypcube[:,:,good_band_start:good_band_stop+1]
wavelengths = wavelengths[good_band_start: good_band_stop+1]
a_ref = hypcube[pix_check[0], pix_check[1], :]

# Check after removing the noisy bands
ax2_f1.plot(wavelengths, a_ref, 'r', linewidth=2, label='Smoothed and removed noisy bands')
ax2_f1.set_xlabel('Wavelengths (nm)', fontsize=12, fontweight='bold')
ax2_f1.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
ax2_f1.legend()
plt.show()


