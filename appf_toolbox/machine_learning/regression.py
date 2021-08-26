from sklearn.externals import joblib
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
            print(samples[i])
            print(wavelengths)
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
                            wavelengths,
                            labels,
                            flag_save=False,
                            flag_fig=False,
                            id_cv=0):
    """
    Modelling a PSL regression using cross-validation.

    :param max_n_components:
    :param num_folds_outer_cv:
    :param num_folds_inner_cv:
    :param input_data_array:
    :param wavelengths: for the purpose of recored only
    :param labels: the values need to be predicted
    :param flag_save:
    :param flag_fig:
    :param id_cv: the id of cv to check
    :return: the record of cv and the model trained using all of the data.

    Author: Huajian Liu
    Email: huajian.liu@adelaide.edu.au

    Version: v0 (10, Feb, 2019)
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

    ####################################################################################################################
    # Outer CV function for computing mean square error compute_mse_pls()
    ####################################################################################################################
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
                  'wavelengths': wavelengths,
                  'input_data_array': input_data_array,
                  'tuned_model_finial': tuned_model_finial
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
        print('The tuned_model_finial and the record has been saved!')

    return record_svr_rbf, tuned_model_finial
    # SVM regression with rbf kernel
################################
# SVM regression with rbf kernel
########################################################################################################################