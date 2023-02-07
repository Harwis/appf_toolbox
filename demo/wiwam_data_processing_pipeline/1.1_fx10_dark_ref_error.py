import sys
sys.path.append('C:/Huajian/python_projects/appf_toolbox_project')
from appf_toolbox.wiwam import wiwam_tools as wt

# Work in Ubuntu but not in Windows

error_data_path = '/media/huajian/TOSHIBA EXT/chicpea_0664_ali_montana/hypdata_0664 chickpea'
error_data_list_path = '//media/huajian/Files/python_projects/chickpea_ali_montana_0664'
error_data_list_file = 'dark_ref_error_report.txt'
good_data_path = '/media/huajian/TOSHIBA EXT/chicpea_0664_ali_montana/hypdata_0664 chickpea'
good_data_name = 'vnir_110_151_9981_2022-11-06_21-18-17'

wt.fix_fx10_dark_error(error_data_path,
                    error_data_list_path, error_data_list_file,
                    good_data_path, good_data_name)
