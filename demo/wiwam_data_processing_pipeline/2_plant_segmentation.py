import sys
sys.path.append('C:/Huajian/python_projects/appf_toolbox_project')
from appf_toolbox.hyper_processing.pre_processing import green_plant_segmentation


# Parameters
hyp_data_path = 'D:/BenRateHemp/hypimg'

model_name_vnir = 'green_crop_seg_model_OneClassSVM_vnir.npy'
model_name_swir = 'green_crop_seg_model_OneClassSVM_swir.npy'
model_path = 'C:/Huajian/python_projects/appf_toolbox_demo_data/crop_segmentation_models'

save_path = 'C:/Huajian/python_projects/ben_hemp_0594/processed_data/segmentation'

green_plant_segmentation(hyp_data_path,
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