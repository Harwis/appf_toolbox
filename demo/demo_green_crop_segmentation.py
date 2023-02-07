import sys
sys.path.append('E:/python_projects/appf_toolbox_project')
import numpy as np
# from matplotlib import pyplot as plt
from appf_toolbox.hyper_processing import envi_funs
from appf_toolbox.hyper_processing import pre_processing as pp
from matplotlib import pyplot as plt


########################################################################################################################
# Parameters
########################################################################################################################
data_path = 'C:/Users/Huajian/OneDrive - University of Adelaide/appf_toolbox_demo_data'
model_path = 'E:/ATP_hypimg_demo/green_crop_segmentation_models'

# data_name = 'vnir_74_104_6235_2019-10-18_01-47-33'
# model_name = 'record_OneClassSVM_vnir_hh_py3.9_sk1.0.2.sav'

data_name = 'swir_74_104_6235_2019-10-18_01-47-33'
model_name = 'record_OneClassSVM_swir_hh_py3.9_sk1.0.2.sav'

band_num_check = 10 # 300 # vnir
pix_check = [186, 183] # [252, 210] # (col, row) vnir

# For smooth filter
# wl = 21 # windows length
# po = 3 # Polyorder

# model_name = 'record_OneClassSVM_vnir_hh_py3.9_sk1.0.2.sav'



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
# Segmentation
########################################################################################################################
bw, pseu = pp.green_plant_segmentation(hypcube, wavelengths, model_path, model_name, flag_check=True)


########
# Check
########
fig, ax = plt.subplots(1, 2)
fig.suptitle(data_name)
ax[0].imshow(hypcube[:, :, band_num_check], cmap='gray')
ax[0].scatter(pix_check[0], pix_check[1], marker='+', color=[1, 0, 0])
ax[0].set_title('Band ' + str(band_num_check) + '@' + str(wavelengths[band_num_check]) + ' nm')
ax[1].plot(wavelengths, hypcube[pix_check[0], pix_check[1], :])
ax[1].set_xlabel('Wavelengths (nm)')
ax[1].set_ylabel('Reflectance')
plt.show()