import numpy as np

def nreai(spectral_signature, wavelength,   blue_edge = [490, 530],  red_edge = [670, 737]):

    """
    Calculate the normalised red-edge area index (NREAI) in the paper of
     Analysis of in situ hyperspectral data for nutrient estimation of giant sequoia
     Gong, P ; Pu, R ; Heald, R. C
     International Journal of Remote Sensing, 01 January 2002, Vol.23(9), pp.1827-1850

    :param spectral_signature: the spectral signature in 1D or 2D ndarray. Each row is a spectrum
    :param wavelength: the wavelength of the corresponding spectra
    :param blue_edge: the first and the last wavelength of the blue edge; default is [490, 530]
    :param red_edge: the first and the last wavelength of the red edge; defaul is [670, 737]
    :return: NREAI values in 1D ndarray
    """

    # If the sppectral_signature is 1D, reshape it to 2D
    if spectral_signature.shape.__len__() == 1:
        spectral_signature = spectral_signature.reshape((1, spectral_signature.shape[0]))


    ind_blue_edge = np.where(np.logical_and(np.logical_or(wavelength > blue_edge[0], wavelength == blue_edge[0]),
                                            np.logical_or(wavelength < blue_edge[1], wavelength == blue_edge[0])))[0]


    ind_red_edge = np.where(np.logical_and(np.logical_or(wavelength > red_edge[0], wavelength == red_edge[0]),
                                           np.logical_or(wavelength < red_edge[1], wavelength == red_edge[0])))[0]

    fir_der = np.delete(spectral_signature, 0, axis=1) - np.delete(spectral_signature, -1, axis=1)

    sum_fir_der_blue = np.sum(fir_der[:, ind_blue_edge], axis=1)
    sum_fir_der_red = np.sum(fir_der[:, ind_red_edge], axis=1)

    nreai = (sum_fir_der_red - sum_fir_der_blue) / (sum_fir_der_red + sum_fir_der_blue + 1e-10)

    return nreai


if __name__ == '__main__':

    import appf_toolbox.hyper_processing.vegetation_indices as vi

    print(vi.nreai.__doc__)

    # Load a data point
    data_path = 'demo_data'
    data_name = 'FieldSpec_demo_data.npy'
    data = np.load(data_path + '/' + data_name)
    data = data.flat[0]
    ref = data['reflectance']
    wavelength = data['wavelength']

    a = nreai(ref, wavelength)


