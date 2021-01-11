"""
my_envi_funs module defines functions to process hyperspectral data in envi format. 
"""

from spectral import *
import numpy as np

def read_hyper_data(path, folder_name):
    """
    Read hyperspectral data in ENVI fromat.

    :param path: the path containing the folders of the data.
    :param folder_name: the name of the data folder.
    :return: {'white:', hypercube_of_white, 'dark:', hypercube_of_dark, 'plant:', hypercube_of_object}, meta_of_plant\

    Author: Huajian liu
    version: v0 (10 May, 2018)
    """
    import spectral.io.envi as envi
    spectral.settings.envi_support_nonlowercase_params = True

    # Reading data
    meta_white = envi.open(path + '/' + folder_name + '/' + 'capture' + '/' + 'WHITEREF_' + folder_name + '.hdr',
                           path + '/' + folder_name + '/' + 'capture' + '/' + 'WHITEREF_' + folder_name + '.raw')

    meta_dark = envi.open(path + '/' + folder_name + '/' + 'capture' + '/' + 'DARKREF_' + folder_name + '.hdr',
                          path + '/' + folder_name + '/' + 'capture' + '/' + 'DARKREF_' + folder_name + '.raw')

    meta_plant = envi.open(path + '/' + folder_name + '/' + 'capture' + '/' + folder_name + '.hdr',
                           path + '/' + folder_name + '/' + 'capture' + '/' + folder_name + '.raw')

    return {'white': meta_white.load(), 'dark': meta_dark.load(), 'plant': meta_plant.load()}, meta_plant


def calibrate_hyper_data(white, dark, plant, trim_rate_t_w=0.05, trim_rate_b_w=0.95):
    """
    Calibrate hyerpsectral data.

    :param white: the hypercube of white returned from read_hyper_data()
    :param dark: the hypercube of dark returned from read_hyper_data()
    :param object: the hypercube of object returned from read_hyper_data()
    :param trim_rate_t_w: the rate for trimming the top of white
    :param trim_rate_b_w: the rate for trmming the bottom of white
    :return: the calibrated hypercube of object in [0 1]

    Author: Huajian liu
    version: v0 (10 May, 2018)
    """
    lines_w = white.shape[0]
    lines_o = plant.shape[0]
    samples = white.shape[1]
    bands = white.shape[2]

    # Take of the ROI of white
    white = white[int(lines_w * trim_rate_t_w):int(lines_w * trim_rate_b_w), 0:samples, 0:bands]

    # Make mean-images of white and dark. The number of lines of mean-images is the same as that of plant.
    white_mean = white.mean(0)
    white_mean = white_mean.reshape(1, samples, bands)
    white_mean = np.tile(white_mean, [lines_o, 1, 1])

    dark_mean = dark.mean(0)
    dark_mean = dark_mean.reshape(1, samples, bands)
    dark_mean = np.tile(dark_mean, [lines_o, 1, 1])

    plant_cal = (plant - dark_mean) / (white_mean - dark_mean + 1e-10)
    plant_cal[plant_cal > 1] = 1
    plant_cal[plant_cal < 0] = 0
    
    return plant_cal
