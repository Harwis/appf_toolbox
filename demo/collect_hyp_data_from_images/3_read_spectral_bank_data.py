import joblib
import numpy as np
from matplotlib import pyplot as plt

########################################################################################################################
# Parameter
########################################################################################################################
band_check = 200
pixel_check = [200, 100] # [row, col]
data_path = 'E:/Data/spectral_bank/fx10eLabScanner'
data_name = 'dark_background_ATP_FX10eLabScanner_22-05-06-16-33-40.sav'


########################################################################################################################
# Load data
########################################################################################################################
data = joblib.load(data_path + '/' + data_name)
wave = data['wavelength']

img_size = data['image size']
ind_obj = data['index of object']
ref = data['reflectance']

########################################################################################################################
# If the data is from an image, reconstruct the image at the selected band using ind_obj
########################################################################################################################
img = np.zeros(img_size)
img[ind_obj] = ref[:, band_check]

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img, cmap='gray')
ax[0].plot(pixel_check[1], pixel_check[0], 'r+')
ax[0].set_title(data['object'] + ' band ' + str(band_check) + '@' + str(wave[band_check]) + 'nm')


########################################################################################################################
# Mean and STD
########################################################################################################################
mean_ref = np.mean(ref, axis=0)
std_ref = np.std(ref, axis=0)
ax[1].plot(wave, mean_ref, linestyle='-', color=[1, 0, 0], label='Mean')
ax[1].plot(wave, mean_ref + std_ref, linestyle='-.', color=[0, 1, 0], label='STD')
ax[1].plot(wave, mean_ref - std_ref, linestyle='-.', color=[0, 1, 0])
ax[1].set_xlabel('Wavelengths (nm)')
ax[1].set_ylabel('Reflectance')


########################################################################################################################
# Plot the reflectance for the selected pixel
########################################################################################################################
ref_id = 0
draw_ref = False
for row, col in zip(data['index of object'][0], data['index of object'][1]):
    if pixel_check[0] == row and pixel_check[1] == col:
        ax[1].plot(wave, ref[ref_id], color=[0, 0, 1], label='row:' + str(row) + ' col:' + str(col))
        draw_ref = True
        break
    else:
        pass

    ref_id = ref_id + 1

if draw_ref == False:
    print('The reflectance of pixel at row:' + str(pixel_check[0]) + ' col:' + str(pixel_check[1]) + ' is not available!')
    print('Please select another pixel of the material.')

plt.legend()
plt.show()