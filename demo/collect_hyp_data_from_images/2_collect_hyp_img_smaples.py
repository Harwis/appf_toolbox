import sys
sys.path.append('E:/python_projects/appf_toolbox_project')
from appf_toolbox.hyper_processing import envi_funs
from appf_toolbox.hyper_processing import pre_processing as pp
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime


########################################################################################################################
# Parameters
########################################################################################################################
object = 'lens culinaris'
mask_colour = np.array([0, 0, 1])
colour_tol = 0.01

data_path = 'E:/Data/0614_ruby'
data_name = 'vnir_103_143_9639_2022-04-13_21-59-25'

mask_path = 'E:/python_projects/ruby_0614/mask_image'
mask_name = 'vnir_103_143_9639_2022-04-13_21-59-25_man.png'

location = 'TPA'
date_collection = '2022-03 to 04'
equipment = 'WIWAM_FX10'
description = 'Ruby_0614'

wavelength_low = 450
wavelength_high = 2400

band_num_check = 400
pix_check = [335, 100] # (col, row)
# For smooth filter
wl = 21 # windows length
po = 3 # Polyorder
flag_save = False


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
ax2_f1.set_title('Smoothing')

########################################################################################################################
# Smooth
########################################################################################################################
hyp_flat = hypcube.reshape((nrows * ncols, nbands), order='C')
hyp_flat = pp.smooth_savgol_filter(hyp_flat, window_length=wl, polyorder=po)
hyp_flat[hyp_flat < 0] = 0
hyp_flat[hyp_flat > 1] = 1
hypcube = hyp_flat.reshape((nrows, ncols, nbands), order='C')
a_ref = hypcube[pix_check[1], pix_check[0], :]

# Check after smoothing
ax2_f1.plot(wavelengths, a_ref, 'r--', label='Smoothed')
ax3_f1 = fig1.add_subplot(2,2,3)
ax3_f1.imshow(hypcube[:, :, band_num_check])
ax3_f1.scatter(pix_check[0], pix_check[1], color='red', marker='+')
ax3_f1.set_title('Band ' + str(band_num_check) + ' ' + str(wavelengths[band_num_check]) + ' nm (After smoothing)' )
ax2_f1.legend()

########################################################################################################################
# Remove noise band
########################################################################################################################
good_band_ind = np.logical_and(
                np.logical_or(wavelengths > wavelength_low, wavelengths == wavelength_low),
                np.logical_or(wavelengths < wavelength_high, wavelengths == wavelength_high))
good_band_ind = np.where(good_band_ind)[0]

hypcube = hypcube[:, :, good_band_ind]
wavelengths = wavelengths[good_band_ind]
a_ref = hypcube[pix_check[1], pix_check[0], :]

# Check after removing the noisy bands
ax4_f1 = fig1.add_subplot(2, 2, 4)
ax4_f1.plot(wavelengths, a_ref, 'r', linewidth=2, label='Smoothed and removed noisy bands')
ax4_f1.set_xlabel('Wavelengths (nm)', fontsize=12, fontweight='bold')
ax4_f1.set_ylabel('Reflectance', fontsize=12, fontweight='bold')
ax4_f1.set_title('Smoothed and removed noisy bands')

########################################################################################################################
# Segment objects
########################################################################################################################
# Make a mask of the object
mask = plt.imread(mask_path + '/' + mask_name)
fig_mask, ax_mask = plt.subplots(1, 2)
ax_mask[0].imshow(mask)

mask = np.logical_and(mask < (mask_colour + colour_tol), mask > (mask_colour - colour_tol))
mask = np.alltrue(mask, axis=2)

ax_mask[1].imshow(mask, cmap='gray')
ax_mask[1].set_title('Mask of ' + object)

# Extract the spectral signatures of the object
ind_obj = np.where(mask)
ref = hypcube[ind_obj]

fig_ref, ax_ref = plt.subplots(1, 1)
fig_ref.suptitle('Check samples')
for i in range(0, ref.shape[0], 100):
    ax_ref.plot(wavelengths, ref[i])
ax_ref.set_xlabel('Wavelengths (nm)')
ax_ref.set_ylabel('Reflectance')


# Save
########################################################################################################################
if flag_save:
    current_time = datetime.now().strftime('%y-%m-%d-%H-%M-%S')

    dict = {'data path': data_path,
            'data name': data_name,
            'location': location,
            'data of collection': date_collection,
            'time of creating the file': current_time,
            'object': object,
            'equipment': equipment,
            'description': description,
            'wavelength': wavelengths,
            'reflectance': ref,
            'image size': mask.shape,
            'index of object': ind_obj}

    import joblib as jl
    jl.dump(dict, object + '_' + location + '_' + equipment + '_' + current_time + '.sav')

print('Data saved!')

plt.show()
