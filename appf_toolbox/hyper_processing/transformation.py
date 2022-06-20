from spectral import *
import numpy as np
from matplotlib import pyplot as plt

def spectral_angle_mapper(hypercube, members, flag_figure=False, member_id=0):
    """
    Call spectral.spectral_angles function. Calculate the spectral angels between the spectral signature of each pixel
    of the hypercube and each of the members (target spectral signature).

    :param hypercube: An MxNXB hypercube. Can be numpy.ndarry or spectral.Image
    :param members: CxB array. Each row is a spectral member.

    :return: sa: MxNxC array of spectral angles.

    Version: v0 (Dec 6, 2019)

    Author: Huajian Liu
    """

    sa = spectral_angles(hypercube, members)

    if flag_figure==True:
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(members[member_id])
        ax1.set_title('Member ' + str(member_id))
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(sa[:, :, member_id])
        ax2.set_title('The spectral angle mapper of member ' + str(member_id))

    return sa


def hc2hhsi(hc):
    """
    Transform an n-dimensional hypercube to hyper-hue, saturation and intensity.

    :param hc: hyperCube(rows x cols x dims) in floating data type of [0 1].
    :return: hh: hypHue (rows x cols x (dims-1)) in floating data type of [0 1].
             s: saturation (rows x cols ) in floating data type of [0 1]
             i: intensity in (rows x cols) in floating data type of [0 1]

    For academic users, please cite:
    Citation 1:
    Liu, H., Lee, S., & Chahl, J.(2017).Transformation of a high-dimensional color space for material classification.
    Journal of the Optical Society of America A, 34(4), 523 - 532, doi: 10.1364/josaa.34.000523.

    Citation 2:
    Liu, H., & Chah, J.S.(2018).A multispectral machine vision system for invertebrate detection on green leaves
    Computer and Elecronics in Agriculture, 150, 279 - 288, doi: https://doi.org/10.1016/j.compag.2018.05.002.

    version 1.1 (Aug 18, 2018)
    Author: Huajian Liu
    """

    import numpy as np

    ####################################################################################################################
    # Calculate the components c
    rows = hc.shape[0]
    cols = hc.shape[1]
    dims = hc.shape[2]

    c = np.zeros((rows, cols, dims-1))
    for i in range(dims - 1):
        nonZeroEle = dims - i # nonZeroEle is the number of non-zero elements of the base unit vector u1, u2, ...
        c[:, :, i] = (nonZeroEle - 1) ** 0.5 / nonZeroEle ** 0.5         * hc[:, :, i] \
                     - 1 / ((nonZeroEle - 1) ** 0.5 * nonZeroEle ** 0.5) * np.sum(hc[:, :, i+1:dims], axis=2)
    ####################################################################################################################

    # Normalise the norms of c to 1 to obtain hyper-hue hh.
    c_norm = np.sum(c ** 2, axis=2) ** 0.5
    c_norm = c_norm + (c_norm == 0) * 1e-10
    c_norm = np.tile(c_norm, (dims - 1, 1, 1))
    c_norm = np.moveaxis(c_norm, 0, -1)
    hh = c / c_norm # add 1e-10 to avoid zero denominators

    # Saturation
    s = hc.max(2) - hc.min(2)
    # s = np.amax(hc, axis=2) - np.amin(hc, axis=2) # The same as above

    # Intensity
    i = 1/dims * np.sum(hc, 2)

    return hh, s, i


def snv(ref, flag_fig=False):
    """
    Calculate the standard normal variate (SNV) of spectral signatures
    :param ref: 1D or 2D ndarray
    :param flag_fig: True or False to show the result
    :return: snv values
    """

    if ref.shape.__len__() == 1:
        ref = ref.reshape((1, ref.shape[0]))

    # meand and std
    mean_ref = np.mean(ref, axis=1)
    std_ref = np.std(ref, axis=1)

    mean_ref = mean_ref.reshape((mean_ref.shape[0], 1), order='C')
    std_ref = std_ref.reshape((std_ref.shape[0], 1), order='C')

    mean_ref = np.tile(mean_ref, (1, ref.shape[1]))
    std_ref = np.tile(std_ref, (1, ref.shape[1]))

    # snv
    snv = (ref - mean_ref) / std_ref

    if flag_fig:
        f = plt.figure()
        a1 = f.add_subplot(1, 2, 1)
        a2 = f.add_subplot(1, 2, 2)

        for a_ref in ref:
            a1.plot(a_ref)
            a1.set_title('Reflectance')

        for a_snv in snv:
            a2.plot(a_snv)
            a2.set_title('SNV')

    return snv


def first_order_derivative(spec_sig, wave=[], flag_check=False, check_interview=10):
    """
    Calculate the 1st-order derivative of spectral singatures.
    :param spec_sig: 1D or 2D np.ndarray. If 1D, it automatically convert it to 2D.
    :param wave: The wavelength of the spectral signatures. Default is [].
    :param flag_check: The flag to check the result or not.
    :param check_interview: If flag_check set to True, it will plot the results from 0 to spec_sig.shape[0] at the
           interview of  check_interview
    :return: The 1st-order derivative of the spectral signatures.

    Author: Huajian Liu
    V0.0
    Data: 6 June, 2022
    """
    import numpy as np

    # If 1D, convert it to 2D
    if spec_sig.shape.__len__() == 1:
        spec_sig = spec_sig.reshape((1, spec_sig.shape[0]))

    # First order derivative
    fir_der = np.delete(spec_sig, obj=0, axis=1) - np.delete(spec_sig, obj= -1, axis=1)

    # Plot
    if flag_check:
        from matplotlib import pyplot as plt

        fig, ax_1der = plt.subplots(1, 1)
        if wave == []:
            for i in range(0, fir_der.shape[0], check_interview):
                ax_1der.plot(fir_der[i])
            ax_1der.set_xlabel('band number', fontsize=12, fontweight='bold')
        else:
            for i in range(0, fir_der.shape[1], check_interview):
                ax_1der.plot(wave, fir_der)
            ax_1der.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        ax_1der.set_ylabel('1st der', fontsize=12, fontweight='bold')
        ax_1der.set_title('First order derivative')

    return fir_der


def second_order_derivative(spec_sig, wave=[], flag_check=False, check_interview=10):
    """
        Calculate the 2end-order derivative of spectral singatures.
        :param spec_sig: 1D or 2D np.ndarray. If 1D, it automatically convert it to 2D.
        :param wave: The wavelength of the spectral signatures. Default is [].
        :param flag_check: The flag to check the result or not.
        :param check_interview: If flag_check set to True, it will plot the results from 0 to spec_sig.shape[0] at the
               interview of  check_interview
        :return: The 2end-order derivative of the spectral signatures.

        Author: Huajian Liu
        V0.0
        Data: 6 June, 2022
        """

    import numpy as np

    # If 1D, convert it to 2D
    if spec_sig.shape.__len__() == 1:
        spec_sig = spec_sig.reshape((1, spec_sig.shape[0]))

    # First order derivative
    fir_der = np.delete(spec_sig, obj=0, axis=1) - np.delete(spec_sig, obj=-1, axis=1)

    # second order derivative
    sec_der = np.delete(fir_der, obj=0, axis=1) - np.delete(fir_der, obj=-1, axis=1)

    if flag_check:
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 1)
        if wave == []:
            for i in range(0, sec_der.shape[0], check_interview):
                ax.plot(sec_der[i])
            ax.set_xlabel('band number', fontsize=12, fontweight='bold')
        else:
            for i in range(0, sec_der.shape[1], check_interview):
                ax.plot(wave, sec_der)
            ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('2end der', fontsize=12, fontweight='bold')
        ax.set_title('Second order derivative')

    return sec_der