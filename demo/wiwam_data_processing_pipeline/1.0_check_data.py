# Check if all project data was downloaded and if there was surplus data downloaded according to
# hypname_barcode
# Check if dark references have errors

import pandas as pd
from os import walk

import sys
sys.path.append('../appf_toolbox')
from appf_toolbox.wiwam import wiwam_tools as wt

########################################################################################################################
# Parameters
########################################################################################################################
hyp_data_path = 'E:/Data/0614_ruby'

barcode_hypname_path = 'C:/Users/Huajian/OneDrive/OD_projects_doc/Rubby_0614'
barcode_hypname_file = 'barcode_hypname.xlsx'
sheet_name = 'barcode_hypname'


########################################################################################################################
# Read barcode_hypname
########################################################################################################################
barcode_hypname = pd.read_excel(barcode_hypname_path + '/' + barcode_hypname_file, sheet_name=sheet_name)
hypnames_vnir = barcode_hypname['vnir']
hypnames_swir = barcode_hypname['swir']
hypnames = pd.concat((hypnames_vnir, hypnames_swir), axis=0)
hypnames = hypnames.to_list()


########################################################################################################################
# Check if all project data has been downloaded
########################################################################################################################
print('Checking if all data in the sheet of ' + sheet_name + ' has been downloaded.')
for (root, dirs, files) in walk(hyp_data_path):
    hypnames_dl = dirs
    break

not_download_count = 0
other_error_count = 0
for i in range(0, hypnames.__len__()):
    if hypnames[i] in hypnames_dl:
        pass
    elif not(hypnames[i] in hypnames_dl):
        not_download_count += 1
        print(hypnames[i] + ' was not downloaded')
    else:
        other_error_count += 1

print('A total of ' + str(not_download_count) + ' data was not downloaded.')
print('A total of other errors (eg. nan) count: ' + str(other_error_count))


########################################################################################################################
# Check if surplus data have been downloaded
########################################################################################################################
print('Checking if surplus data was downloaded.')
error_count = 0
for i in range(0, hypnames_dl.__len__()):
    if hypnames_dl[i] in hypnames:
        pass
    else:
        error_count += 1
        print(hypnames_dl[i] + ' is surplus data')
print('A total of ' + str(error_count) + ' data is surplus data')


########################################################################################################################
# Check dark reference error
########################################################################################################################
error_count = 0
# <<<<<<< HEAD
print('Checking dark reference error......')
# =======
# list_error_data = []
# >>>>>>> d6f0b1cd5160b226e6b478783bd9749d57034435
for i in range(0, hypnames_dl.__len__()):
    if hypnames_dl[i][0:4] == 'swir':
        pass
    else:
        flag_error = wt.check_fx10_dark_error(hyp_data_path, hypnames_dl[i], threshold=500)
        if flag_error == 1:
            error_count += 1
            # list_error_data.append(hypnames_dl[i])
            print(hypnames_dl[i] + ' has dark reference error.')
print('A total of ' + str(error_count) + ' data have dark reference error')
# print(list_error_data)

