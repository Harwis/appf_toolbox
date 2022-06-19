import sys
sys.path.append('E:/python_projects/appf_toolbox_project')
from appf_toolbox.hyper_processing.pre_processing import green_plant_segmentation_batch


# Parameters
hyp_data_path = 'E:/Data/ruby_0614'

model_name_vnir = 'record_OneClassSVM_vnir_hh.sav'
model_name_swir = 'record_OneClassSVM_swir_hh.sav'
# model_path = '/media/huajian/Files/python_projects/appf_toolbox_demo_data/green_seg_model_20210129'
model_path = 'E:/python_projects/p12_green_segmentation/model_ruby_0614'

save_path = 'E:/Data/ruby_0614_segmentation_v2'

gamma = 0.6

green_plant_segmentation_batch(hyp_data_path,
                               model_name_vnir=model_name_vnir,
                               model_name_swir=model_name_swir,
                               model_path=model_path,
                               gamma=0.8,
                               flag_remove_noise=True,
                               white_offset_top=0.1,
                               white_offset_bottom=0.9,
                               # band_R=10,
                               # band_G=50,
                               # band_B=150,
                               flag_save=True,
                               save_path=save_path)