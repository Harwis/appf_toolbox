# Check if all project data was downloaded and if there was surplus data downloaded according to
# hypname_barcode

import pandas as pd
from os import walk

import sys
sys.path.append('../appf_toolbox')
from appf_toolbox.wiwam import wiwam_tools as wt

########################################################################################################################
# Parameters
########################################################################################################################
hyp_data_path = 'D:/BenRateHemp/hypimg'

barcode_hypname_path = 'C:/Huajian\python_projects/ben_hemp_0594/processed_data'
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

error_count = 0
nan_count = 0
for i in range(0, hypnames.__len__()):
    if hypnames[i] in hypnames_dl:
        pass
    elif pd.isnull(hypnames[i]):
        nan_count += 1
    else:
        error_count += 1
        print(hypnames[i] + ' was not downloaded')
print('A total of ' + str(error_count) + ' data was not downloaded.')
print('A total of ' + str(nan_count) + ' nan')


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
        print(hypnames_dl[i] + ' in is surplus data')
print('A total of ' + str(error_count) + ' data is surplus data')


########################################################################################################################
# Check dark reference error
########################################################################################################################
error_count = 0
for i in range(0, hypnames_dl.__len__()):
    print('Checking ' + hypnames_dl[i] + '.')
    if hypnames_dl[i][0:4] == 'swir':
        pass
    else:
        flag_error = wt.check_fx10_dark_error(hyp_data_path, hypnames_dl[i], threshold=500)
        if flag_error == 1:
            error_count += 1
            print(hypnames_dl[i] + ' has dark reference error.')
print('A total of ' + str(error_count) + ' data have dark reference error')

