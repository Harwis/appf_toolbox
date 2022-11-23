import joblib
# import joblib
import optunity.metrics
import optunity
import datetime
from sklearn.cross_decomposition import PLSRegression
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.svm import SVR


########################################################################################################################
# Calculate prediction errors
def errors_prediction(labels, predictions):
    """
    Calculate typical errors of the predictions of a regression model.

    :param labels: The ground-truth values
    :param predictions: The predicted values
    :return: A dictionary of the typical regression errors

    Author: Huajian Liu
    Email: huajian.liu@adelaide.edu.au

    Version: v0 (10, Apr, 2019)
    """

    errors = {'r2_score': r2_score(labels, predictions),
              'bias': np.mean(predictions - labels),
              'mean_absolute_error': mean_absolute_error(labels, predictions),
              'median_absolute_error': median_absolute_error(labels, predictions),
              'rmse': mean_squared_error(labels, predictions) ** 0.5,
              'mse': mean_squared_error(labels, predictions),
              }

    return errors
########################################################################################################################


########################################################################################################################
# Calculate average errors
def errors_average(error_each_fold):
    """Calculates the average errors of cross-validation returned from errors_prediction()"""
    # The subscribe "test" should be "validation"
    sum_r2_score_train = 0
    sum_bias_train = 0
    sum_mean_absolute_error_train = 0
    sum_median_absolute_error_train = 0
    sum_mse_train = 0
    sum_rmse_train = 0

    sum_r2_score_test = 0
    sum_bias_test = 0
    sum_mean_absolute_error_test = 0
    sum_median_absolute_error_test = 0
    sum_mse_test = 0
    sum_rmse_test = 0

    for a_record in error_each_fold:
        sum_r2_score_train += a_record['errors_train']['r2_score']
        sum_bias_train += a_record['errors_train']['bias']
        sum_mean_absolute_error_train += a_record['errors_train']['mean_absolute_error']
        sum_median_absolute_error_train += a_record['errors_train']['median_absolute_error']
        sum_mse_train += a_record['errors_train']['mse']
        sum_rmse_train += a_record['errors_train']['rmse']

        sum_r2_score_test += a_record['errors_test']['r2_score']
        sum_bias_test += a_record['errors_test']['bias']
        sum_mean_absolute_error_test += a_record['errors_test']['mean_absolute_error']
        sum_median_absolute_error_test += a_record['errors_test']['median_absolute_error']
        sum_mse_test += a_record['errors_test']['mse']
        sum_rmse_test += a_record['errors_test']['rmse']

    ave_r2_score_train = sum_r2_score_train/error_each_fold.__len__()
    ave_bias_train = sum_bias_train/error_each_fold.__len__()
    ave_mean_absolute_error_train = sum_mean_absolute_error_train/error_each_fold.__len__()
    ave_median_absolute_error_train = sum_median_absolute_error_train/error_each_fold.__len__()
    ave_mse_train = sum_mse_train / error_each_fold.__len__()
    ave_rmse_train = sum_rmse_train / error_each_fold.__len__()

    ave_r2_score_test = sum_r2_score_test / error_each_fold.__len__()
    ave_bias_test = sum_bias_test / error_each_fold.__len__()
    ave_mean_absolute_error_test = sum_mean_absolute_error_test / error_each_fold.__len__()
    ave_median_absolute_error_test = sum_median_absolute_error_test / error_each_fold.__len__()
    ave_mse_test = sum_mse_test / error_each_fold.__len__()
    ave_rmse_test = sum_rmse_test / error_each_fold.__len__()

    ave_errors = {'ave_r2_score_train': ave_r2_score_train,
                  'ave_bias_train': ave_bias_train,
                  'ave_mean_absolute_error_train': ave_mean_absolute_error_train,
                  'ave_median_absolute_error_train': ave_median_absolute_error_train,
                  'ave_mse_train': ave_mse_train,
                  'ave_rmse_train': ave_rmse_train,
                  'ave_r2_score_test': ave_r2_score_test,
                  'ave_bias_test': ave_bias_test,
                  'ave_mean_absolute_error_test': ave_mean_absolute_error_test,
                  'ave_median_absolute_error_test': ave_median_absolute_error_test,
                  'ave_mse_test': ave_mse_test,
                  'ave_rmse_test': ave_rmse_test}

    return ave_errors
########################################################################################################################


########################################################################################################################
# Plot the result of regression
# # Check a result of prediction.
def plot_regression_result(lab, pre):
    from matplotlib import pyplot as plt

    # Plot the points and lines
    polyfit_para = np.polyfit(lab, pre, 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(pre, lab, c='red', edgecolors='k')
    ax.plot(polyfit_para[1] + polyfit_para[0] * lab, lab, c='blue', linewidth=1)
    ax.plot(lab, lab, color='green', linewidth=1)
    ax.grid(True)
    plt.xlabel('Predicted values', fontsize=12, fontweight='bold')
    plt.ylabel('Labeled values', fontsize=12, fontweight='bold')
    plt.title('Regression errors', fontsize=14, fontweight='bold')

    # Calculate errors
    reg_errors = errors_prediction(lab, pre)

    # Print errors
    rangey = max(lab) - min(lab)
    rangex = max(pre) - min(pre)
    plt.text(min(pre) + 0.02 * rangex, max(lab) - 0.1  * rangey, 'R$^{2}=$ %5.3f' % reg_errors['r2_score'])
    plt.text(min(pre) + 0.02 * rangex, max(lab) - 0.15 * rangey, 'RMSE: %5.3f' % reg_errors['rmse'])
    plt.text(min(pre) + 0.02 * rangex, max(lab) - 0.2 * rangey, 'Bias: %5.3f' % reg_errors['bias'])
    plt.text(min(pre) + 0.02 * rangex, max(lab) - 0.25 * rangey, 'MeanABS: %5.3f' % reg_errors['mean_absolute_error'])
    plt.text(min(pre) + 0.02 * rangex, max(lab) - 0.3  * rangey, 'MedianABS: %5.3f' % reg_errors['median_absolute_error'])
########################################################################################################################


########################################################################################################################
def plot_samples_with_colourbar(samples, labels, wavelengths, input_type='Input data values', title='Title'):
    """
    Plot the samples (reflectance values) with a colour bar which is defined by the values of the labels.

    :param samples: input data array; usually reflectance values
    :param labels: the values of the labels (the parameter need to be measured)
    :param wavelengths: the wavelengths of the reflectance; 1D array; if samples are not reflectance, set it to []
    :param input_type: reflectance, pca, etc
    :param title: title for plot
    :return: return 0 if no errors
    """
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    # Make color map.
    norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    if wavelengths == []:
        plt.figure()
        plt.title(title)
        for i in range(samples.shape[0]):
            plt.plot(samples[i], c=cmap.to_rgba(labels[i]), alpha=1)
        plt.colorbar(cmap,
                     ticks=np.linspace(np.round(labels.min(), decimals=3), np.round(labels.max(), decimals=3), 10))
        plt.xlabel('Dimensions of ' + input_type, fontsize=12, fontweight='bold')
        plt.ylabel(input_type, fontsize=12, fontweight='bold')
    else:
        plt.figure()
        plt.title(title)
        for i in range(samples.shape[0]):
            plt.plot(wavelengths, samples[i], c=cmap.to_rgba(labels[i]), alpha=1)
        plt.colorbar(cmap,
                     ticks=np.linspace(np.round(labels.min(), decimals=3), np.round(labels.max(), decimals=3), 10))
        plt.xlabel('Wavelengths (nm)', fontsize=12, fontweight='bold')
        plt.ylabel(input_type, fontsize=12, fontweight='bold')
########################################################################################################################


########################################################################################################################
def print_ave_errors_cv(ave_errors):
    print('')
    print('The average errors of CV of training is:')
    print('r^2_train: ', ave_errors['ave_r2_score_train'])
    print('rmse_train: ', ave_errors['ave_rmse_train'])
    print('bias train: ', ave_errors['ave_bias_train'])
    print('mean_absolute_error_train: ', ave_errors['ave_mean_absolute_error_train'])
    print('median_absolute_error_train: ', ave_errors['ave_median_absolute_error_train'])
    print('')
    print('The average errors of CV of validation is: ')
    print('r^2_validation: ', ave_errors['ave_r2_score_test'])
    print('rmse_validation: ', ave_errors['ave_rmse_test'])
    print('bias validation: ', ave_errors['ave_bias_test'])
    print('mean_absolute_error_validation: ', ave_errors['ave_mean_absolute_error_test'])
    print('median_absolute_error_validation: ', ave_errors['ave_median_absolute_error_test'])
    print('')
    return 0
########################################################################################################################


########################################################################################################################
# Find the optimal_n_components
########################################################################################################################
def find_optimal_n_components_plsr(x_train, y_train, max_n_components, num_folds_cv):
    # Store the average mse of different n_components
    list_ave_mse = []
    for n_components in range(1, max_n_components + 1):
        # Store the mse of the current n_components in cv
        list_mse = []

        # Define the function of compute mse in the training and testing data
        def compute_mse(x_train, y_train, x_test, y_test):
            model = PLSRegression(n_components=n_components).fit(x_train, y_train)
            predictions = model.predict(x_test)
            mse = optunity.metrics.mse(y_test, predictions)
            list_mse.append(mse)
            return mse

        # The cv object
        cv = optunity.cross_validated(x=x_train, y=y_train, num_folds=num_folds_cv)
        try:
            compute_mse_cv = cv(compute_mse)
            compute_mse_cv()
        except ValueError:
            print('Value error. The n_component in PLSR is bigger than the dimension of the input data!')
            print('Found the optimal n_component in the valid range.')
            break

        # Record the ave_mes for this parameter
        ave_mse = np.mean(list_mse)
        list_ave_mse.append(ave_mse)

    # Find the min and index of list_ave_mse
    optimal_n_components = np.argmin(list_ave_mse) + 1
    print("The optimal number of components of PLS: ", optimal_n_components)

    return optimal_n_components


########################################################################################################################
# modelling_PLSRegression
########################################################################################################################
def modelling_PLSRegression(max_n_components,
                            num_folds_outer_cv,
                            num_folds_inner_cv,
                            input_data_array,
                            labels,
                            note='',
                            flag_save=False,
                            flag_fig=False,
                            id_cv=0):
    """
    Modelling a PSL regression using cross-validation.

    :param max_n_components:
    :param num_folds_outer_cv:
    :param num_folds_inner_cv:
    :param input_data_array:
    :param labels: the values need to be predicted
    :param note: some note for training the model.
    :param flag_save:
    :param flag_fig:
    :param id_cv: the id of cv to check
    :return: the record of cv and the model trained using all of the data.

    Author: Huajian Liu
    Email: huajian.liu@adelaide.edu.au

    Version: v0.0 (10, Feb, 2019)
             v0.1 (26, Aug, 2022) Input of "wavelength was removed"; Input of "note" was added.
    """
    start = datetime.datetime.now()
    print('')
    print('PLS regression')
    print('The range of n_components is: [1, ' + str(max_n_components) + ']')
    print('')

    # For records
    date_time = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    save_record_name = 'record_plsr_' + date_time + '.sav'
    save_model_name = 'model_plsr' + date_time + '.sav'

    ######################################################################
    # Outer CV function for computing mean square error compute_mse_pls()
    ######################################################################
    print('Conducting outer cross-validation')

    # For record.
    params_each_fold = []
    errors_each_fold = []
    predictions_labels_each_fold = []
    tuned_models_each_fold = []

    # Define the function for outer CV
    def compute_mse_pls(x_train, y_train, x_test, y_test):
        """Find the optimized n_nomponents.
           Train a model using the opt-parameter.
           compute MSE
        """

        # ##############################################################################################################
        # # Find the optimal parameter (n_components) of PLS
        # ##############################################################################################################
        optimal_n_components = find_optimal_n_components_plsr(x_train, y_train, max_n_components=max_n_components,
                                                              num_folds_cv=num_folds_inner_cv)


        ################################################################################################################
        # Train a model using the optimal parameters and the x_train and y_train
        ################################################################################################################
        # Train
        tuned_model = PLSRegression(n_components=optimal_n_components).fit(x_train, y_train)

        # Predict the testing data and training data
        predictions_train = tuned_model.predict(x_train)
        predictions_train = predictions_train.reshape(x_train.shape[0], order='C') # Make it one-D
        predictions_test = tuned_model.predict(x_test)
        predictions_test = predictions_test.reshape(x_test.shape[0], order='C')

        ################################################################################################################
        # Record errors and parameters
        ################################################################################################################
        errors_train = errors_prediction(y_train, predictions_train)
        errors_test = errors_prediction(y_test, predictions_test)
        print('R^2_train: ', errors_train['r2_score'])
        print('R^2_validation:', errors_test['r2_score'])
        print('')

        predictions_labels_each_fold.append({'predictions_train': predictions_train,
                                             'labels_train': y_train,
                                             'predictions_test': predictions_test,
                                             'labels_test': y_test})
        params_each_fold.append({'optimal_n_component': optimal_n_components})
        errors_each_fold.append({'errors_train': errors_train, 'errors_test': errors_test})
        tuned_models_each_fold.append(tuned_model)

        return errors_test['mse']

    # Activate outer CV
    outer_cv = optunity.cross_validated(x=input_data_array, y=labels, num_folds=num_folds_outer_cv)
    compute_mse_pls = outer_cv(compute_mse_pls)
    compute_mse_pls()

    print('The cross-validation has been done!', datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    stop = datetime.datetime.now()
    print('Total time used ', stop - start)

    ave_errors = errors_average(errors_each_fold)
    print_ave_errors_cv(ave_errors)

    ################################################################################################################
    # Train a model using all of the data
    ################################################################################################################
    print('')
    print('Traing the finial model using all of the data')

    optimal_n_components = find_optimal_n_components_plsr(input_data_array, labels, max_n_components=max_n_components,
                                                          num_folds_cv=num_folds_outer_cv)

    # Train a model using the optimal parameters and the x_train and y_train
    tuned_model_finial = PLSRegression(n_components=optimal_n_components).fit(input_data_array, labels)
    print('')

    ####################################################################################################################
    # Record the results
    ####################################################################################################################
    record_pls = {'model_name': save_model_name,
                  'date_time': date_time,
                  'num_folds_outer_cv': num_folds_outer_cv,
                  'num_folds_inner_cv': num_folds_inner_cv,
                  'tuned_models_each_fold': tuned_models_each_fold,
                  'predictions_labels_each_fold': predictions_labels_each_fold,
                  'optimal_parameters_each_fold': params_each_fold,
                  'errors_each_fold': errors_each_fold,
                  'average_errors': ave_errors,
                  'num_samples': input_data_array.shape[0],
                  'tuned_model_final': tuned_model_finial,
                  'note': note
                  }

    if flag_fig:
        # Plot a record in one (random selected) of the cv
        plot_regression_result(predictions_labels_each_fold[id_cv]['labels_train'],
                               predictions_labels_each_fold[id_cv]['predictions_train'])
        plot_regression_result(predictions_labels_each_fold[id_cv]['labels_test'],
                               predictions_labels_each_fold[id_cv]['predictions_test'])



    ####################################################################################################################
    # Save record
    ####################################################################################################################
    if flag_save:
        joblib.dump(record_pls, save_record_name)
        print('The the record has been saved in the current working folder.')

    return record_pls




########################################################################################################################
# SVM regression with rbf kernel
################################
def modelling_svr_rbf(C_svr_rbf,
                      gamma_svr_rbf,
                      wavelengths_range,
                      input_type,
                      num_folds_outer_cv,
                      num_iter_inner_cv,
                      num_folds_inner_cv,
                      num_evals_inner_cv,
                      samples,
                      wavelengths,
                      labels,
                      flag_save,
                      flag_fig):
    """ Model a svr with rbf kernel."""

    start = datetime.datetime.now()
    print('')
    print('svr (kernel = rbf)')
    print('The range of C is: ', C_svr_rbf)
    print('The range of gamma is: ', gamma_svr_rbf)
    print('')

    # For records
    date_time = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    model_name = 'svr_rbf'
    save_record_name = 'record' + '_' + wavelengths_range + '_' + input_type + '_' + model_name + '.sav'
    save_model_name = 'model' + '_' + wavelengths_range + '_' + input_type + '_' + model_name + '.sav'

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # CV
    # ///
    print('Conducting cross-validation')

    # For record
    params_each_fold = []
    errors_each_fold = []
    predictions_labels_each_fold = []
    tuned_models_each_fold = []

    # ==================================================================================================================
    # Thf function for outer_cv
    # ==========================
    def compute_mse_svr_rbf(x_train, y_train, x_test, y_test):
        """Find the optimal hyperparameters of svm;
           Train a model using the optmal parametes
           compute MSE
        """

        # -------------------------------------------------------------------------------------------------------------
        # Find optimal parameters
        # ------------------------
        @optunity.cross_validated(x=x_train, y=y_train, num_iter=num_iter_inner_cv,
                                  num_folds=num_folds_inner_cv)
        def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
            model = SVR(C=C, gamma=gamma).fit(x_train, y_train)
            predictions = model.predict(x_test)
            return optunity.metrics.mse(y_test, predictions)

        # Optimise parameters
        optimal_pars, _, _ = optunity.minimize(tune_cv, num_evals=num_evals_inner_cv, C=C_svr_rbf, gamma=gamma_svr_rbf)
        print("THe optimal hyperparameters of SVR (kernel = rbf): " + str(optimal_pars))
        # -----------------------
        # Find optimal parameters
        # -------------------------------------------------------------------------------------------------------------

        # Train a model using the optimal parameters and the x_train and y_train
        tuned_model = SVR(**optimal_pars).fit(x_train, y_train)

        # Predict the testing data and training data
        predictions_train = tuned_model.predict(x_train)
        predictions_train = predictions_train.reshape(x_train.shape[0], order='C') # Make it one-D
        predictions_test = tuned_model.predict(x_test)
        predictions_test = predictions_test.reshape(x_test.shape[0], order='C')

        # Errors
        errors_train = errors_prediction(y_train, predictions_train)
        errors_test = errors_prediction(y_test, predictions_test)
        print('R^2_train: ', errors_train['r2_score'])
        print('R^2_test:', errors_test['r2_score'])

        # Save the parameters and errors
        predictions_labels_each_fold.append({'predictions_train': predictions_train,
                                             'labels_train': y_train,
                                             'predictions_test': predictions_test,
                                             'labels_test': y_test})
        params_each_fold.append(optimal_pars)
        errors_each_fold.append({'errors_train': errors_train, 'errors_test': errors_test})
        tuned_models_each_fold.append(tuned_model)
        return errors_test['mse']
    # =========================
    # The function for outer cv
    # ==================================================================================================================

    # The fellow is the same as:
    # @optunity.cross_validated(x=samples, y=labels, num_folds=num_folds_outer_cv)
    # def compute_mse_svr_rbf:
    #     ...
    #
    # compute_mse_svr_rbf()
    outer_cv = optunity.cross_validated(x=samples, y=labels, num_folds=num_folds_outer_cv)  # function decoter
    compute_mse_svr_rbf = outer_cv(compute_mse_svr_rbf) # Decorate computer_mse_svr_rbf
    compute_mse_svr_rbf()

    print('The cross-validation has been done!', datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    stop = datetime.datetime.now()
    print('Total time used ', stop - start)
    # ///
    # CV
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Record the results
    ave_errors = errors_average(errors_each_fold)
    record_svr_rbf = {'model_name': save_model_name,
                      'date_time': date_time,
                      'C_range': C_svr_rbf,
                      'gamma_range': gamma_svr_rbf,
                      'num_folds_outer_cv': num_folds_outer_cv,
                      'num_iter_inner_cv': num_iter_inner_cv,
                      'num_folds_inner_cv': num_folds_inner_cv,
                      'num_evals_inner_cv': num_evals_inner_cv,
                      'tuned_models_each_fold': tuned_models_each_fold,
                      'predictions_labels_each_fold': predictions_labels_each_fold,
                      'optimal_parameters_each_fold': params_each_fold,
                      'errors_each_fold': errors_each_fold,
                      'average_errors': ave_errors,
                      'wavelengths': wavelengths
                      }

    # Print average of cv
    print_ave_errors_cv(ave_errors)

    if flag_fig:
        # Plot a record in one (random selected) of the cv
        plot_regression_result(predictions_labels_each_fold[0]['labels_train'],
                               predictions_labels_each_fold[0]['predictions_train'])
        plot_regression_result(predictions_labels_each_fold[0]['labels_test'],
                               predictions_labels_each_fold[0]['predictions_test'])


    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # Train a model using all of the data
    # ////////////////////////////////////
    # ==================================================================================================================
    # Find the optimal parameters
    # ============================
    print('Training a SVR (kernel = rbf) instance.')
    @optunity.cross_validated(x=samples, y=labels, num_iter=num_iter_inner_cv,
                              num_folds=num_folds_inner_cv)
    def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
        model = SVR(C=C, gamma=gamma).fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)


    # Optimise parameters
    optimal_pars, _, _ = optunity.minimize(tune_cv, num_evals=num_evals_inner_cv, C=C_svr_rbf, gamma=gamma_svr_rbf)
    # ============================
    # Find the optimal parameters
    # ==================================================================================================================

    # Train a model using all of the data
    tuned_model_finial = SVR(**optimal_pars).fit(samples, labels)
    # ///////////////////////////////////
    # Train a model using all of the data
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Save the model
    if flag_save:
        joblib.dump(record_svr_rbf, save_record_name)
        joblib.dump(tuned_model_finial, save_model_name)
        print('The tuned_model_final and the record has been saved!')

    return record_svr_rbf, tuned_model_finial
    # SVM regression with rbf kernel
################################
# SVM regression with rbf kernel
###################################

#####################
# Make regression map
#####################
def make_regression_map(hyp_data,
                        wavelength,
                        green_seg_model_path,
                        green_seg_model_name,
                        regression_model_path,
                        regression_model_name,
                        val_min,
                        val_max,
                        band_r=97,
                        band_g=52,
                        band_b=14,
                        gamma=0.7,
                        roi =[],
                        rotation = 0,
                        flag_gaussian_filter=True,
                        radius_dilation=3,
                        sigma_gaussian=5,
                        flag_figure=False,
                        flag_remove_noise=True,
                        flag_remove_border=False,
                        selem_size=3):
    """
    Make a 2D regression map of a 3D hypercube based on trained crop segmentation model and regression model.
    :param hyp_data: A calibrated 3D hypercube of row x col x dim. Float format in the range of [0, 1]
    :param wavelength: The corresponding wavelengths of the hypercube.
    :param green_seg_model_path: The path to save the crop segmentation model.
    :param green_seg_model_name: The name of the crop segmentation model.
    :param regression_model_path: The path of the regression model.
    :param regression_model_name: The name of the regression model.
    :param val_min: the minimum value of the regression results.
    :param val_max: the maxum value of the regresson results.
    :param band_r: The red band number to create a pseudo RGB image. Default is 97.
    :param band_g: The green band number to create a pseudo RGB image. Default is 52.
    :param band_b: The blue band number to create a pseddo RGB image. Default is 14.
    :param gamma: The gamma value for exposure adjustment. Default is 0.7
    :param roi: Region-of interest give as a dictionary {row_top, row_bottom, column_left, column_right}. Default is [].
    :param rotation: Rotate the images. Default is 0 degree.
    :param flag_gaussian_filter: The flage to apply Gaussian filter or not.
    :param radius_dilation: The radius for object dilation.
    :param sigma_gaussian: Sigma value for gaussian operation. Default is 5.
    :param flag_figure: The flat to show the result or not.
    :param flag_remove_noise: The flat to remove noise or not in the crop-segmented image.
    :param flag_remove_border: For crop segmentation. The flag to remove the borders of the crops. The size of the
           border is determined by selem_size. Default is False.
    :param selem_size: For crop segmentation. If flag_remove_border set to True, erosion will be conducted using selem
           np.ones((selem_size, selem_size)). Default is 3.
    :return: A dictionary contain the results of crop segmentation and the map.

    Version 1.0
    Data: Aug 25 2022
    Author: Huajian Liu
    """

    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent)
    from appf_toolbox.hyper_processing import transformation as tf
    from appf_toolbox.hyper_processing import pre_processing as pp
    from appf_toolbox.image_processing import gray_img_processing as gip
    from matplotlib import pyplot as plt
    import joblib as joblib
    from sklearn.decomposition import PCA
    import matplotlib as mpl
    from skimage import filters

    # -------------------
    # Region of interest.
    # -------------------
    if roi == []:
        pass
    else:
        hyp_data = hyp_data[roi['row_top']:roi['row_bottom'], roi['column_left']:roi['column_right']]

    # ---------
    # Rotation
    # ---------
    if rotation == 0:
        pass
    else:
        hyp_data = pp.rotate_hypercube(hyp_data, rotation)

    # ------------------
    # Crop segmentation
    # ------------------
    print('Conducting crop segmentation......')
    bw_crop, pseu_crop = pp.green_plant_segmentation(hyp_data,
                                                     wavelength,
                                                     green_seg_model_path,
                                                     green_seg_model_name,
                                                     band_R=band_r,
                                                     band_G=band_g,
                                                     band_B=band_b,
                                                     gamma=gamma,
                                                     flag_remove_noise=flag_remove_noise,
                                                     flag_check=flag_figure)



    # -----------
    # Load regression mode
    # ----------
    print('Creating maps......')
    regression_model = joblib.load(regression_model_path + '/' + regression_model_name)

    # ------
    # Smooth
    # ------
    (n_row, n_col, _) = hyp_data.shape
    hyp_data = hyp_data.reshape(n_row * n_col, hyp_data.shape[2], order='c')
    hyp_data = pp.smooth_savgol_filter(hyp_data,
                                       window_length=regression_model['note']['smooth_window_length'],
                                       polyorder=regression_model['note']['smooth_window_polyorder'])
    hyp_data[hyp_data < 0] = 0

    # Resampling
    hyp_data = pp.spectral_resample(hyp_data, wavelength, regression_model['note']['wavelengths used in the model'])
    n_band = hyp_data.shape[1]

    # ----------------
    # Transformation
    # ----------------
    if regression_model['note']['data_transformation'] == 'none':
        print('Input data is reflectance.')
        input = hyp_data
    elif regression_model['note']['data_transformation'] == 'pca':
        print('Input data is PCA n_pc = ' + str(regression_model['Note']['number_components_pca']))
        pca = PCA(n_components=regression_model['Note']['number_components_pca'])
        pcs = pca.fit_transform(hyp_data)
        print('Explained variance: ')
        print(pca.explained_variance_)
        print('Explained variance ratio: ')
        print(pca.explained_variance_ratio_)
        print('Cumulative explained variance ratio: ')
        print(pca.explained_variance_ratio_.cumsum())

        input = pcs

    elif regression_model['note']['data_transformation'] == 'hyp-hue' or \
         regression_model['note']['data_transformation'] == 'hyper-hue' or \
         regression_model['note']['data_transformation'] == 'hh':
        print('Input data is hyp-hue')
        hyp_data = hyp_data.reshape(n_row, n_col, n_band)
        hyp_data, _, _ = tf.hc2hhsi(hyp_data)  # after this, hyp_data is hyp_hue (Inplace operation to save memory).
        input = hyp_data.reshape(n_row * n_col, n_band - 1)

    elif regression_model['note']['data_transformation'] == 'snv':
        print('Input data is SNV')
        input = tf.snv(hyp_data)

    else:
        print('Wrong transformation method!')

    # -----------
    # Make map
    # ----------
    # Predict
    regression_map = regression_model['tuned_model_final'].predict(input)
    regression_map = regression_map.reshape((n_row, n_col))
    regression_map[regression_map < val_min] = val_min
    regression_map[regression_map > val_max] = val_max

    if flag_gaussian_filter:
        # Dilation
        regression_map = gip.obj_dilation_2d(bw_crop==1, regression_map, r=radius_dilation, flag_check=flag_figure)

        # # Gaussian filter
        regression_map = filters.gaussian(regression_map, sigma=sigma_gaussian, mode='nearest', cval=0)

    # Set background to 0
    regression_map[bw_crop == 0] = 0

    # Convert regression map to rgb image with white background.
    plt.imsave('regression_map_rgb.png', regression_map, cmap='jet', vmin=val_min, vmax=val_max)
    regression_map_rgb = plt.imread('regression_map_rgb.png')

    # Set the bk to white
    regression_map_rgb[bw_crop == 0] = [1, 1, 1, 1]

    # ---------------
    # Show the result
    # ---------------
    if flag_figure:
        # Show the pseudo RGB image of original data
        wave_r = wavelength[band_r]
        wave_g = wavelength[band_g]
        wave_b = wavelength[band_b]
        fig1, ax_fig1 = plt.subplots(1, 3)
        ax_fig1[0].imshow(pseu_crop)
        ax_fig1[0].set_title('Pseudo RGB R=' + str(wave_r) + ' G=' + str(wave_g) + ' B=' + str(wave_b))

        # Show the regression map
        ax_fig1[1].imshow(regression_map, cmap='jet', vmin=val_min, vmax=val_max)
        ax_fig1[1].set_title('Regression map')
        ax_fig1[2].imshow(regression_map_rgb)
        ax_fig1[2].set_title('Regression map (RGB)')

        # Make a colour bar
        norm = mpl.colors.Normalize(vmin=val_min, vmax=val_max)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap='jet')
        cmap.set_array([])
        fig1.colorbar(mappable=cmap, shrink=1)

    return {'bw_crop': bw_crop,
            'pseu_crop': pseu_crop,
            'regression_map': regression_map,
            'regression_map_rgb': regression_map_rgb
            }


#############################
# Demo of make_regression map
#############################
if __name__ == "__main__":
    import sys
    import numpy as np
    sys.path.append('E:/python_projects/appf_toolbox_project')
    from appf_toolbox.hyper_processing import envi_funs

    # Input parameters
    data_path = 'E:\Data\wheat_n_0493\wiw_20191017'
    data_name = 'vnir_74_104_6125_2019-10-17_00-44-10'

    seg_model_path = 'E:/python_projects/p12_green_segmentation/green_seg_model_20220201/wiwam_py3.9'
    seg_model_name = 'record_OneClassSVM_vnir_hh_py3.9_sk1.0.2.sav'

    regression_model_path = 'E:/python_projects/wheat_n_0493/models'
    regression_model_name = 'record_plsr_22-09-09-19-38-21_vnir_hyper-hue_py3.9.sav'

    val_min = 1
    val_max = 7
    roi = {'row_top': 100,  'row_bottom': 400, 'column_left': 150, 'column_right': 350}
    rotation = 180

    # Read the data
    raw_data, meta_plant = envi_funs.read_hyper_data(data_path, data_name)

    wavelengths = np.zeros((meta_plant.metadata['Wavelength'].__len__(),))
    for i in range(wavelengths.size):
        wavelengths[i] = float(meta_plant.metadata['Wavelength'][i])

    # Calibrate the data
    hypcube = envi_funs.calibrate_hyper_data(raw_data['white'], raw_data['dark'], raw_data['plant'],
                                             trim_rate_t_w=0.1, trim_rate_b_w=0.95)

    map = make_regression_map(hypcube,
                            wavelengths,
                            seg_model_path,
                            seg_model_name,
                            regression_model_path,
                            regression_model_name,
                            val_min,
                            val_max,
                            band_r=97,
                            band_g=52,
                            band_b=14,
                            gamma=0.7,
                            roi = roi,
                            rotation=rotation,
                            flag_gaussian_filter=True,
                            radius_dilation=3,
                            sigma_gaussian=3,
                            flag_figure=True,
                            flag_remove_noise=True,
                            flag_remove_border=False,
                            selem_size=3)

