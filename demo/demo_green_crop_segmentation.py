import sys
sys.path.append('/media/huajian/Files/python_projects/appf_toolbox_project')
import numpy as np
# from matplotlib import pyplot as plt
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
# wl = 21 # windows length
# po = 3 # Polyorder

model_name_vnir = 'green_crop_seg_model_OneClassSVM_vnir.npy'
# model_name_swir = 'green_crop_seg_model_OneClassSVM_swir.npy'
model_path = '/media/huajian/Files/python_projects/appf_toolbox_demo_data/green_seg_model_20210129'

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
bw, pseu = pp.green_plant_segmentation(hypcube, wavelengths, model_path, model_name_vnir, flag_check=True)