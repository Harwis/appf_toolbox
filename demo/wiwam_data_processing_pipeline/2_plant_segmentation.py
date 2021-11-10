import sys
sys.path.append('C:/Huajian/python_projects/appf_toolbox_project')
from appf_toolbox.hyper_processing.pre_processing import green_plant_segmentation_batch


# Parameters
hyp_data_path = '/media/huajian/Files/Data/registration_vnir_swir/registration_vnir_swir'

model_name_vnir = 'green_crop_seg_model_OneClassSVM_vnir.npy'
model_name_swir = 'green_crop_seg_model_OneClassSVM_swir.npy'
model_path = '/media/huajian/Files/python_projects/appf_toolbox_demo_data/green_seg_model_20210129'

save_path = '/media/huajian/Files/python_projects'

green_plant_segmentation_batch(hyp_data_path,
                               model_name_vnir=model_name_vnir,
                               model_name_swir=model_name_swir,
                               model_path=model_path,
                               gamma=0.8,
                               flag_remove_noise=True,
                               white_offset_top=0.1,
                               white_offset_bottom=0.9,
                               band_R=10,
                               band_G=50,
                               band_B=150,
                               flag_save=True,
                               save_path=save_path)