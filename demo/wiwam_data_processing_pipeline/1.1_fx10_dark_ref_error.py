import sys
sys.path.append('C:/Huajian/python_projects/appf_toolbox_project')
from appf_toolbox.wiwam import wiwam_tools as wt


error_data_path = '/media/huajian/TOSHIBA EXT/crown_rot_0590_top_view'
error_data_list_path = '/media/huajian/Files/python_projects/crown_rot_0590/crown_rot_0590_processed_data/crown_rot_0590_top_view'
error_data_list_file = 'dark_ref_error_report'
good_data_path = '/media/huajian/TOSHIBA EXT/crown_rot_0590_top_view'
good_data_name = 'vnir_100_140_9108_2021-08-03_04-29-08'

wt.fix_fx10_dark_error(error_data_path,
                    error_data_list_path, error_data_list_file,
                    good_data_path, good_data_name)
