def obj_dilation_2d(mask, img, r=3, flag_check=False):
    """
    Dilate the objects in a gray-scal image.
    :param mask: Bool type. True for objects and False for background.
    :param img: The gray-gray scale image for dilation.
    :param r: the radius of the sliding window.
    :param flag_check: Flag to check the result.
    :return: The dilated image.

    v0.0 Aug 26, 2022
    Author: Huajian Liu
    """
    import numpy as np
    from matplotlib import pyplot as plt

    if flag_check:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Original image')
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title('Mask')

    nrows = mask.shape[0]
    ncols = mask.shape[1]
    for row in range(nrows):
        for col in range(ncols):

            if np.all(
                    mask[row - r:row + r + 1, col - r:col + r + 1] == False):  # If the window on the background
                pass
            elif row - r < 0 or row + r > (nrows - 1) or col - r < 0 or col + r > (
                    ncols - 1):  # If this window is out of the image
                pass
            elif np.all(
                    mask[row - r:row + r + 1, col - r:col + r + 1] == True):  # If the winddow are on the objects
                pass
            elif mask[row, col] == False:  # The window covers the object and background and the pixel is background
                # The indices of objects in the window
                ind_obj_local = np.where(mask[row - r:row + r + 1, col - r:col + r + 1] == True)
                row_obj_global = ind_obj_local[0] + row - r
                col_obj_global = ind_obj_local[1] + col - r

                # The mean values of the object inside the window.
                win_mean = np.mean(img[row_obj_global, col_obj_global])

                # Copy the mean value to the background in the window
                img[row, col] = win_mean


    if flag_check:
        ax[2].imshow(img, cmap='gray')
        ax[2].set_title('After 2D dilation')
        plt.show()

    return img

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    mask = plt.imread('E:/python_projects/appf_toolbox_demo_data/swir_100_140_9063_2021-08-31_01-06-01_bw.png')
    mask = mask==1
    mask = mask[:, :, 0]
    img = plt.imread('E:/python_projects/appf_toolbox_demo_data/swir_100_140_9063_2021-08-31_01-06-01.png')
    img_dia = obj_dilation_2d(mask, img, r=3, flag_check=True)

