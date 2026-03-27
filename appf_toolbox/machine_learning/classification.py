########################################################################################################################
def plot_samples_with_colourbar(samples, labels, wavelengths=[], input_type='Input data values', title='Title'):
    """
    Plot the samples with a colour bar which is defined by the values of the labels.

    :param samples: input data array; usually reflectance values
    :param labels: the values of the labels. E.g. 0, 1, 2 ......
    :param x_axis_value: the x_axis_value in the plot; default is [] which will lead to x axis values of 1, 2, 3 ......
    :param input_type: A string of "reflectance", "pca", etc
    :param title: title for plot
    :return: return 0 if no errors

    Version 1.0 Date: Aug 25, 2021
    Author: Huajian Liu huajian.liu@adelaide.edu.au
    """

    from matplotlib import pyplot as plt
    import matplotlib as mpl
    import numpy as np

    uni_label = np.unique(labels)
    cmap = plt.get_cmap('jet', len(uni_label))
    norm = mpl.colors.BoundaryNorm(np.arange(len(uni_label) + 1)-0.5, len(uni_label))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if wavelengths == []:
        plt.figure()
        plt.title(title)
        for i in range(samples.shape[0]):
            ind_color = np.where(uni_label == labels[i])[0][0]
            plt.plot(samples[i], c=cmap(ind_color))
        plt.xlabel('Dimensions of ' + input_type, fontsize=12, fontweight='bold')
    else:
        plt.figure()
        plt.title(title)
        for i in range(samples.shape[0]):
            ind_color = np.where(uni_label == labels[i])[0][0]
            plt.plot(wavelengths, samples[i], c=cmap(ind_color))
        plt.xlabel('Wavelengths (nm)', fontsize=12, fontweight='bold')

    plt.colorbar(sm, ticks=uni_label)
    plt.ylabel(input_type, fontsize=12, fontweight='bold')


def average_bi_classification_metrics(record_cv):
    """
    Average the binary classification metrics returned by binary_classification_report() and pr_report 
    in CV. 
    binary_classification_report() returns: 
 
    :param recrod_cv: a list of binary classification metrics. 

    returne: The average binary classification metrics of the CV.  
    """
    import numpy as np

    # For recording average
    # Standard 
    ave_con_mat = []
    ave_accuracy = []
    ave_precision = []
    ave_recall = []
    ave_f1 = []
    ave_kappa = []

    # pr-curve
    ave_pr_auc = []
    ave_best_th = []
    
    # Best threshold
    ave_con_mat_bt = []
    ave_accuracy_bt = []
    ave_precision_bt = []
    ave_recall_bt = []
    ave_f1_bt = []
    ave_kappa_bt = []

    
    # Load the cv report
    rep1 = 'standard_bi_classification_report'
    rep2 = 'pr_curve_report'
    rep3 = 'best_threshold_report'
    for a_record in record_cv:
        # Standard
        ave_con_mat.append(a_record[rep1]['confusion_matrix'])
        ave_accuracy.append(a_record[rep1]['accuracy'])
        ave_precision.append(a_record[rep1]['precision'])
        ave_recall.append(a_record[rep1]['recall'])
        ave_f1.append(a_record[rep1]['f1'])
        ave_kappa.append(a_record[rep1]['kappa'])

        # pr-curve
        ave_pr_auc.append(a_record[rep2]['pr_auc'])
        ave_best_th.append(a_record[rep2]['best_threshold'])
        
        # Best threshold
        ave_con_mat_bt.append(a_record[rep3]['confusion_matrix'])
        ave_accuracy_bt.append(a_record[rep3]['accuracy'])
        ave_precision_bt.append(a_record[rep3]['precision'])
        ave_recall_bt.append(a_record[rep3]['recall'])
        ave_f1_bt.append(a_record[rep3]['f1'])
        ave_kappa_bt.append(a_record[rep3]['kappa'])
        
    # ------    
    # Standard
    # ------   
    # Average confusion matrix
    ave_con_mat = np.asarray(ave_con_mat)
    ave_con_mat = np.mean(ave_con_mat, axis=0)
    ave_con_mat = np.round(ave_con_mat).astype(int)

    # Average accuracy
    ave_accuracy = np.asarray(ave_accuracy)
    ave_accuracy = np.mean(ave_accuracy, axis=0)
    ave_accuracy = np.round(ave_accuracy, 4)

    # Average precision
    ave_precision = np.asarray(ave_precision)
    ave_precision = np.mean(ave_precision, axis=0)
    ave_precision = np.round(ave_precision, 4)

    # Average recall
    ave_recall = np.asarray(ave_recall)
    ave_recall = np.mean(ave_recall, axis=0)
    ave_recall = np.round(ave_recall, 4)

    # Average f1 score
    ave_f1 = np.asarray(ave_f1)
    ave_f1 = np.mean(ave_f1, axis=0)
    ave_f1 = np.round(ave_f1, 4)

    # Average kappa
    ave_kappa = np.asarray(ave_kappa)
    ave_kappa = np.mean(ave_kappa, axis=0)
    ave_kappa = np.round(ave_kappa, 4)

    # ------
    # pr-curve
    # ------
    # Average pr_auc
    ave_pr_auc = np.asarray(ave_pr_auc)
    ave_pr_auc = np.mean(ave_pr_auc, axis=0)
    ave_pr_auc = np.round(ave_pr_auc, 4)

    # Average best threshold
    ave_best_th = np.asarray(ave_best_th)
    ave_best_th = np.mean(ave_best_th, axis=0)
    ave_best_th = np.round(ave_best_th, 4)

    # ------
    # Best threshold
    # ------
    # Confusion matrix
    ave_con_mat_bt = np.asarray(ave_con_mat_bt)
    ave_con_mat_bt = np.mean(ave_con_mat_bt, axis=0)
    ave_con_mat_bt = np.round(ave_con_mat_bt).astype(int)

    # Average accuracy
    ave_accuracy_bt = np.asarray(ave_accuracy_bt)
    ave_accuracy_bt = np.mean(ave_accuracy_bt, axis=0)
    ave_accuracy_bt = np.round(ave_accuracy_bt, 4)

    # Average precision
    ave_precision_bt = np.asarray(ave_precision_bt)
    ave_precision_bt = np.mean(ave_precision_bt, axis=0)
    ave_precision_bt = np.round(ave_precision_bt, 4)

    # Average recall
    ave_recall_bt = np.asarray(ave_recall_bt)
    ave_recall_bt = np.mean(ave_recall_bt, axis=0)
    ave_recall_bt = np.round(ave_recall_bt, 4)

    # Average f1 score
    ave_f1_bt = np.asarray(ave_f1_bt)
    ave_f1_bt = np.mean(ave_f1_bt, axis=0)
    ave_f1_bt = np.round(ave_f1_bt, 4)

    # Average kappa
    ave_kappa_bt = np.asarray(ave_kappa_bt)
    ave_kappa_bt = np.mean(ave_kappa_bt, axis=0)
    ave_kappa_bt = np.round(ave_kappa_bt, 4)
    
    # Save the results to a dictionary
    ave_metrics =  {
        # Standard
        'confusion_matrix': str(ave_con_mat),
        "accuracy": ave_accuracy,
        "precision": ave_precision,
        "recall": ave_recall,
        "f1": ave_f1,
        "kappa": ave_kappa,

        # pr-curve
        "pr_auc": ave_pr_auc,
        'best_threshold': ave_best_th,
        
        # Best threshod
        'confusion_matrix_best_th': str(ave_con_mat_bt),
        "accuracy_best_th": ave_accuracy_bt,
        "precision_best_th": ave_precision_bt,
        "recall_best_th": ave_recall_bt,
        "f1_best_th": ave_f1_bt,
        "kappa_best_th": ave_kappa_bt
        }

    return ave_metrics


def average_classification_metrics(record_cv, average_method='macro avg'):
    """
    HL: 20260307
    Need revision 

    Average classification metrics for multiple classes. Designed for repeated_k-fold_cv.
    :param record_cv: recored of cross validaton retruned from
    :param average_method: 'macro avg' or 'weighted avg'
    :return: Averaged classification metrics.
    """
    import numpy as np

    # For recording average
    ave_con_mat_tra = []
    ave_precision_tra = []
    ave_recall_tra = []
    ave_accuracy_tra = []
    ave_f1_tra = []

    ave_con_mat_val = []
    ave_precision_val = []
    ave_recall_val = []
    ave_accuracy_val = []
    ave_f1_val = []

    for a_record in record_cv:
        # Prepare data for averaging
        ave_con_mat_tra.append(a_record['Confusion matrix of train'])
        ave_accuracy_tra.append(a_record['Classification report of train']['accuracy'])
        ave_recall_tra.append(a_record['Classification report of train'][average_method]['recall'])
        ave_precision_tra.append(a_record['Classification report of train'][average_method]['precision'])
        ave_f1_tra.append(a_record['Classification report of train'][average_method]['f1-score'])

        ave_con_mat_val.append(a_record['Confusion matrix of validation'])
        ave_accuracy_val.append(a_record['Classification report of validation']['accuracy'])
        ave_recall_val.append(a_record['Classification report of validation'][average_method]['recall'])
        ave_precision_val.append(a_record['Classification report of validation'][average_method]['precision'])
        ave_f1_val.append(a_record['Classification report of validation'][average_method]['f1-score'])

    # Average confusion matrix
    ave_con_mat_tra = np.asarray(ave_con_mat_tra)
    ave_con_mat_tra = np.mean(ave_con_mat_tra, axis=0)
    ave_con_mat_tra = np.round(ave_con_mat_tra).astype(int)

    ave_con_mat_val = np.asarray(ave_con_mat_val)
    ave_con_mat_val = np.mean(ave_con_mat_val, axis=0)
    ave_con_mat_val = np.round(ave_con_mat_val).astype(int)

    # Average accuracy
    ave_accuracy_tra = np.asarray(ave_accuracy_tra)
    ave_accuracy_tra = np.mean(ave_accuracy_tra, axis=0)
    ave_accuracy_tra = np.round(ave_accuracy_tra, 4)

    ave_accuracy_val = np.asarray(ave_accuracy_val)
    ave_accuracy_val = np.mean(ave_accuracy_val, axis=0)
    ave_accuracy_val = np.round(ave_accuracy_val, 4)

    # Average recall
    ave_recall_tra = np.asarray(ave_recall_tra)
    ave_recall_tra = np.mean(ave_recall_tra, axis=0)
    ave_recall_tra = np.round(ave_recall_tra, 4)

    ave_recall_val = np.asarray(ave_recall_val)
    ave_recall_val = np.mean(ave_recall_val, axis=0)
    ave_recall_val = np.round(ave_recall_val, 4)

    # Average precision
    ave_precision_tra = np.asarray(ave_precision_tra)
    ave_precision_tra = np.mean(ave_precision_tra, axis=0)
    ave_precision_tra = np.round(ave_precision_tra, 4)

    ave_precision_val = np.asarray(ave_precision_val)
    ave_precision_val = np.mean(ave_precision_val, axis=0)
    ave_precision_val = np.round(ave_precision_val, 4)

    # Average f1 score
    ave_f1_tra = np.asarray(ave_f1_tra)
    ave_f1_tra = np.mean(ave_f1_tra, axis=0)
    ave_f1_tra = np.round(ave_f1_tra, 4)

    ave_f1_val = np.asarray(ave_f1_val)
    ave_f1_val = np.mean(ave_f1_val, axis=0)
    ave_f1_val = np.round(ave_f1_val, 4)

    ave_metrics = {'average_metrics_train': {'conf_mat': str(ave_con_mat_tra),
                                         'accuracy': ave_accuracy_tra,
                                         'recall': ave_recall_tra,
                                         'precision': ave_precision_tra,
                                         'f1': ave_f1_tra},
                   'average_metrics_validation': {'conf_mat': str(ave_con_mat_val),
                                              'accuracy': ave_accuracy_val,
                                              'recall': ave_recall_val,
                                              'precision': ave_precision_val,
                                              'f1': ave_f1_val}}

    return ave_metrics



def repeadted_kfold_cv(input, label, n_splits, n_repeats, tune_model, karg, random_state=0,
                       average_method='macro avg', # mul-classification
                       note='',
                       input_name='x',
                       label_name='y',
                       flag_save=False,
                       file_name_prefix='cv_record'):
    """
    Perform repeated k-folds cross validation of classification.

    :param input: Input data in the format of 2D numpy array.
    :param label: The ground-trued labels. 1D numpy array in int.
    :param n_splits: The number of splits for cross validation
    :param n_repeats: The number of repeat for cross validation.
    :param tune_model: The function for tuning the models.
    :param karg: Key words arguments for tune_model()
    :param random_state: Random state for cross-validation. Default is 0
    :param flag_save: Flag to save the record. If set to True, it will save the record as a .save file in the present
           working directory. Default is False
    :param file_name_prefix: The file name to save the record. Default is 'cv_record'.
    :return: If have valid record, it returns a dictionary recording the report of repeated cross validation; otherwise
           it return -1.

     Version 1.0 Date: Aug 25, 2021 Tested for binary classification.
     Author: Huajian Liu huajian.liu@adelaide.edu.au
    """

    from sklearn.model_selection import RepeatedKFold
    import numpy as np
    import joblib
    from datetime import datetime
    import sklearn.metrics as met
    import os
    import pandas as pd

    # Performing repeated cross validation
    # First, calculate the number of samples of each class; total samples
    classes, counts = np.unique(label, return_counts=True)
    total_samples = label.shape[0]

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    record_each_cv_tra = []
    record_each_cv_val = []
    count_cv = 1
    for train_ind, val_index in rkf.split(input):
        print('')
        print('========== cross validation ' + str(count_cv) + ' ==========')

        # Number of samples in each class
        classes_val, counts_val = np.unique(label[val_index], return_counts=True)
        classes_tra, counts_tra = np.unique(label[train_ind], return_counts=True)

        # The classes of training or validation should be the same of the total classes
        if classes_tra.shape[0] != classes.shape[0]:
            print('The classes of training is ', classes_tra)
            print('Do not take into account in cv.')
        elif classes_val.shape[0] != classes.shape[0]:
            print("The classes of validation is ", classes_val)
            print('Do not take into account in cv.')
        else:
            # Tune a model
            try:
                tuned_model = tune_model(input[train_ind], label[train_ind], **karg)
                print('Tuned model:')
                print(tuned_model)
            except ValueError as e:
                print(e)
                print('Cannot tune model ')
                count_cv += 1
                continue

            # Prediction uisng the default score threshold = 0.5
            output_tra = tuned_model.predict(input[train_ind])
            output_val = tuned_model.predict(input[val_index])
            
            # Predict the scores
            y_score_tra = tuned_model.predict_proba(input[train_ind])[:, 1]
            y_score_val = tuned_model.predict_proba(input[val_index])[:, 1]
                          
            
            if classes.shape[0] > 2:

                # Output dict for the average_metrics function
                report_tra = met.classification_report(label[train_ind], output_tra, output_dict=True, zero_division=0)
                report_val = met.classification_report(label[val_index], output_val, output_dict=True, zero_division=0)

                # Confusion matrix
                conf_max_tra = met.confusion_matrix(label[train_ind], output_tra)
                conf_max_val = met.confusion_matrix(label[val_index], output_val)

           
                print('')
                print('--- Train ---')
                print('Classes', classes_tra)
                print('Count of each class:')
                print(counts_tra)
                print('Confusion matrix:')
                print(conf_max_tra)
                print('roc_auc:', roc_auc_tra)
                print('Metrics')
                print(met.classification_report(label[train_ind], output_tra, zero_division=0))

                print('')
                print('--- Validation ---')
                print('Classes', classes_val)
                print('Count of each class:')
                print(counts_val)
                print('Confusion matrix:')
                print(conf_max_val)
                print('roc_auc:', roc_auc_val)
                print(met.classification_report(label[val_index], output_val, zero_division=0))

                # Record
                record_each_cv.append({'Classification report of train': report_tra,
                                    'Count of train': counts_tra,
                                    'Classes train': classes_tra,
                                    'Confusion matrix of train': conf_max_tra,
                                    ''

                                    'Classification report of validation': report_val,
                                    'Count of validation': counts_val,
                                    'Classes validation': classes_val,
                                    'Confusion matrix of validation': conf_max_val,
                                    })
            else: # Binary classification
                # On training data
                print('\nReprot of train in CV')
                std_bi_class_report_tra = binary_classification_report(
                                                    y_true = label[train_ind],
                                                    y_pred = output_tra,
                                                    pos_label=1,
                                                    print_report = True,
                                                    )

                best_th_report_tra, pr_curve_report_tra = pr_curve_report(
                                    y_true = label[train_ind],
                                    y_score = y_score_tra,
                                    pos_label=1,
                                    strategy = "f1")
                                    

                # On validation data
                print('\nReprot of validation in CV')
                std_bi_class_report_val = binary_classification_report(
                                                    y_true = label[val_index],
                                                    y_pred = output_val,
                                                    pos_label=1,
                                                    print_report = True,
                                                    )
                best_th_report_val, pr_curve_report_val = pr_curve_report(
                                                    y_true = label[val_index],
                                                    y_score = y_score_val,
                                                    pos_label=1,
                                                    strategy = "f1")

                # Record of each CV
                record_each_cv_tra.append({'standard_bi_classification_report': std_bi_class_report_tra,
                                           'pr_curve_report': pr_curve_report_tra,
                                           'best_threshold_report': best_th_report_tra
                                           })  
                                                                                        
                record_each_cv_val.append({'standard_bi_classification_report': std_bi_class_report_val,
                                           'pr_curve_report': pr_curve_report_val,
                                           'best_threshold_report': best_th_report_val
                                           })

        count_cv += 1

    if record_each_cv_tra.__len__() == 0:
        print('No valid record in cross-validation.')
        return -1

    # Average cv metrics and train a final model using all of the data
    final_model = tune_model(input, label, **karg)
    y_hat = final_model.predict(input)
    y_score = final_model.predict_proba(input)[:, 1]

    if classes.shape[0] > 2:
        print('Average method of mul-classification :' + average_method)
        ave_metrics = average_classification_metrics(record_each_cv, average_method=average_method)
    else:
        average_metrics_train = average_bi_classification_metrics(record_each_cv_tra)
        average_metrics_validation = average_bi_classification_metrics(record_each_cv_val)
        
        std_final_model_report = binary_classification_report(
                        y_true = label,
                        y_pred = y_hat,
                        pos_label=1,
                        print_report = False,
                        )

        best_th_final_model_report,  pr_curve_final_model_report = pr_curve_report(
                                                    y_true = label,
                                                    y_score = y_score,
                                                    pos_label=1,
                                                    strategy = "f1")

    # Print the summary of cross-validation
    print()
    print('====================================================')
    print('Summary of ' + str(n_splits) + '-fold cross validation with ' + str(n_repeats) + ' repeats')
    print('====================================================')
    print('Total samples: ', total_samples)
    print('Classes:', classes)
    print('Count in each class: ', counts)
    print('')
    print('Average metrics of training data:')
    print(pd.DataFrame(average_metrics_train, index=[0]))
    print('\nAverage metrics of validation data')
    print(pd.DataFrame(average_metrics_validation, index=[0]))
  
   
    # Record
    record = {'stand_final_model_report': std_final_model_report,
              'best_th_final_model_report': best_th_final_model_report, 
              'pr_curve_final_model_report': pr_curve_final_model_report,
              'final_model': final_model,  
              'average_metrics_train': average_metrics_train, 
              'average_metrics_validation': average_metrics_validation,
              'total_samples': total_samples,
              'classes': str(classes),
              'count_in_each_class': str(counts),            
              'note': note,
              'input': input,
              'input_name': input_name,
              'label': label,
              'label_name': label_name}

    # Save
    if flag_save:
        file_name_prefix = file_name_prefix + '_' + datetime.now().strftime('%y-%m-%d-%H-%M-%S') + '.sav'
        joblib.dump(record, file_name_prefix)
        print('CV record saved at ' +  os.getcwd() + '/' + file_name_prefix)

    return record


def tune_svm_classification(input,
                            label,
                            svm_kernel='rbf',
                            svm_c_range=[1, 100],
                            svm_gamma_range=[1, 50],
                            svm_tol=1e-3,
                            class_weight='None',
                            opt_num_iter_cv=5,
                            opt_num_fold_cv=5,
                            opt_num_evals=5,
                            flag_output_probability=True):
    """
    Tune a support vector machine classificaition model based on sklearn.svm.SVC
    :param input: The input data for training the model. 2D numpy array
    :param label: The ground-trued labels for training the model. 1D numpy array in int.
    :param svm_kernel: The kernel function of SVM. Refer to sklearn.svm.SVC
    :param svm_c_range: The searching range of C of sklearn.svm.SVC. Default is [1, 100]
    :param svm_gamma_range: The searching range of gamma of sklearn.svm.SVC. Defaul is [1, 50]
    :param svm_tol: The tol value of sklearn.svm.SVC.
    :param opt_num_iter_cv: The number of iteration of cross validation of optunity.
    :param opt_num_fold_cv: The number of fold of cross validation of optunity.
    :param opt_num_evals: The number of evaluation of optunity.
    :return: A tuned SVM model for classification.

    Author: Huajian Liu email: huajian.liu@adelaide.edu.au
    Version: 1.0 Date: August 20, 2021
    """
    from sklearn.svm import SVC
    import optunity.metrics

    # Search the optimal parameters
    @optunity.cross_validated(x=input, y=label, num_iter=opt_num_iter_cv, num_folds=opt_num_fold_cv)
    def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
        model = SVC(kernel=svm_kernel, C=C, gamma=gamma, tol=svm_tol, class_weight=class_weight, probability=flag_output_probability)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.error_rate(y_test, predictions) # error_rate = 1.0 - accuracy(y, yhat)

    optimal_pars, _, _ = optunity.minimize(tune_cv, num_evals=opt_num_evals, C=svm_c_range, gamma=svm_gamma_range)

    # Train a model using the optimal parameters
    tuned_model = SVC(kernel=svm_kernel, **optimal_pars, tol=svm_tol, class_weight=class_weight, probability=True).fit(input, label)
    
    return tuned_model


def tune_nn_classification(input,
                           label,
                           nn_hidden_layer_size,
                           nn_solver,
                           nn_alpha_range,
                           nn_max_iter=200,
                           nn_random_state=1,
                           opt_num_iter_cv=5,
                           opt_num_fold_cv=5,
                           opt_num_evals=5):
    """
    Tune a nn classification model based on sklearn.neural_network
    :param input: The input data for training the model. 2D numpy array
    :param label: The ground-trued labels for training the model. 1D numpy array in int.
    :param opt_num_iter_cv: The number of iteration of cross validation of optunity.
    :param opt_num_fold_cv: The number of fold of cross validation of optunity.
    :param opt_num_evals: The number of evaluation of optunity.
    :return: A tuned nn model for classification.

    Author: Huajian Liu email: huajian.liu@adelaide.edu.au
    Version: 1.0 Date: March 6, 2024
    """
    from sklearn.neural_network import MLPClassifier
    import optunity.metrics

    # Search the optimal parameters
    @optunity.cross_validated(x=input, y=label, num_iter=opt_num_iter_cv, num_folds=opt_num_fold_cv)
    def tune_cv(x_train, y_train, x_test, y_test, alpha):
        model = MLPClassifier(solver=nn_solver, hidden_layer_sizes=nn_hidden_layer_size, max_iter=nn_max_iter, random_state=nn_random_state,
                              alpha=alpha) # need to be tuned
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.error_rate(y_test, predictions) # error_rate = 1.0 - accuracy(y, yhat)

    optimal_pars, _, _ = optunity.minimize(tune_cv, num_evals=opt_num_evals, alpha=nn_alpha_range)

    # Train a model using the optimal parameters
    tuned_model = MLPClassifier(solver=nn_solver, hidden_layer_sizes=nn_hidden_layer_size, random_state=nn_random_state,
                                **optimal_pars).fit(input, label).fit(input, label)
    return tuned_model


def tune_rf_classification(input,
                           label,
                           rf_max_depth=2,
                           rf_random_state=1,
                           ):
    """
    Tune a random forest classification model based on sklearn.neural_network
    :param input: The input data for training the model. 2D numpy array
    :param label: The ground-trued labels for training the model. 1D numpy array in int.
    :return: A tuned rf model for classification.

    Author: Huajian Liu email: huajian.liu@adelaide.edu.au
    Version: 1.0 Date: March 7, 2024
    """
    from sklearn.ensemble import RandomForestClassifier

    # Train a model using the optimal parameters
    tuned_model = RandomForestClassifier(max_depth=rf_max_depth, random_state=rf_random_state).fit(input, label)
    return tuned_model


def tune_gaussianNB_classification(input,
                              label,
                              ):
    """
    Tune a Naive Bayes Classifier. As Naive Bayes classifier does not need to tune any parameter, we do nothing here,
    just keeping a standard format as other classifiers.

    :param input: The input data for training the model. 2D numpy array
    :param label: The ground-trued labels for training the model. 1D numpy array in int.
    :return: A tuned model for classification.

    Author: Huajian Liu email: huajian.liu@adelaide.edu.au
    Version: 1.0 Date: Aug 21, 2024
    """
    from sklearn.naive_bayes import GaussianNB 

    # Train a model using the optimal parameters
    tuned_model = GaussianNB().fit(input, label)
    return tuned_model






def make_class_map(hyp_data,
                   wavelength,
                   seg_model_path,
                   seg_model_name,
                   classification_model_path,
                   classification_model_name,
                   band_r=97,
                   band_g=52,
                   band_b=14,
                   gamma=0.7,
                   flag_remove_noise=True,
                   flag_figure=False,
                   flag_remove_border=False,
                   selem_size=3):

    """
    Make a 2D classification map of a 3D hypercube based on trained crop segmentation model and classification model.

    :param hyp_data: A calibrated 3D hypercube of row x col x dim. Float format in the range of [0, 1]
    :param wavelength: The corresponding wavelengths of the hypercube.
    :param seg_model_path: The path to save the crop segmentation model.
    :param seg_model_name: The name of the crop segmentation model.
    :param classification_model_path: The path of the classification model.
    :param classification_model_name: The name of the classification model.
    :param band_r: The red band number to create a pseudo RGB image. Default is 97.
    :param band_g: The green band number to create a pseudo RGB image. Default is 52.
    :param band_b: The blue band number to create a pseddo RGB image. Default is 14.
    :param gamma: The gamma value for exposure adjustment. Default is 0.7
    :param flag_remove_noise: The flat to remove noise or not in the crop-segmented image.
    :param flag_figure: The flat to show the result or not.
    :param flag_remove_border: For crop segmentation. The flag to remove the borders of the crops. The size of the
           border is determined by selem_size. Default is False.
    :param selem_size: For crop segmentation. If flag_remove_border set to True, erosion will be conducted using selem
           np.ones((selem_size, selem_size)). Default is 3.
    :return: A dictionary contain the results of crop segmentation and the map.

    Version 0 Only support 2C-classification
    Data: Feb 14 2022
    Author: Huajian Liu
    """

    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent)
    from appf_toolbox.hyper_processing import transformation as tf
    from appf_toolbox.hyper_processing import pre_processing as pp
    from matplotlib import pyplot as plt
    import joblib as joblib
    import numpy as np
    from skimage import exposure
    from sklearn.decomposition import PCA

    # ------------------------------------------------------------------------------------------------------------------
    # Crop segmentation
    # ------------------------------------------------------------------------------------------------------------------
    print('Conducting crop segmentation......')
    bw_crop, pseu_crop = pp.green_plant_segmentation(hyp_data,
                                                     wavelength,
                                                     seg_model_path,
                                                     seg_model_name,
                                                     band_R=band_r,
                                                     band_G=band_g,
                                                     band_B=band_b,
                                                     gamma=gamma,
                                                     flag_remove_noise=flag_remove_noise,
                                                     flag_check=flag_figure)

    # ------------------------------------------------------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------------------------------------------------------
    print('Creating maps......')
    # Load model
    classification_model = joblib.load(classification_model_path + '/' + classification_model_name)

    # ------------------------------------------------------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------------------------------------------------------
    # Save RGB for showing results
    r = hyp_data[:, :, band_r].copy()
    g = hyp_data[:, :, band_g].copy()
    b = hyp_data[:, :, band_b].copy()
    wave_r = wavelength[band_r]
    wave_g = wavelength[band_g]
    wave_b = wavelength[band_b]

    # Smooth
    (n_row, n_col, _) = hyp_data.shape
    hyp_data = hyp_data.reshape(n_row * n_col, hyp_data.shape[2], order='c')
    hyp_data = pp.smooth_savgol_filter(hyp_data,
                                       window_length=classification_model['Note']['smooth_window_length'],
                                       polyorder=classification_model['Note']['smooth_window_polyorder'])
    hyp_data[hyp_data < 0] = 0

    # Resampling
    hyp_data = pp.spectral_resample(hyp_data, wavelength, classification_model['Note']['wavelengths used in the model'])
    n_band = hyp_data.shape[1]

    # ------------------------------------------------------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------------------------------------------------------
    if classification_model['Note']['data_transformation'] == 'none':
        print('Input data is reflectance.')
        input = hyp_data
    elif classification_model['Note']['data_transformation'] == 'pca':
        print('Input data is PCA n_pc = ' + str(classification_model['Note']['number_components_pca']))
        pca = PCA(n_components=classification_model['Note']['number_components_pca'])
        pcs = pca.fit_transform(hyp_data)
        print('Explained variance: ')
        print(pca.explained_variance_)
        print('Explained variance ratio: ')
        print(pca.explained_variance_ratio_)
        print('Cumulative explained variance ratio: ')
        print(pca.explained_variance_ratio_.cumsum())

        input = pcs

    elif classification_model['Note']['data_transformation'] == 'hyp-hue':
        print('Input data is hyp-hue')
        hyp_data = hyp_data.reshape(n_row, n_col, n_band)
        hyp_data, _, _ = tf.hc2hhsi(hyp_data) # after this, hyp_data is hyp_hue (Inplace operation to save memory).
        input = hyp_data.reshape(n_row * n_col, n_band - 1)

    elif classification_model['Note']['data_transformation'] == 'snv':
        print('Input data is SNV')
        input = tf.snv(hyp_data)

    else:
        print('Wrong transformation method!')

    # ------------------------------------------------------------------------------------------------------------------
    # Make map
    # ------------------------------------------------------------------------------------------------------------------
    # Predict
    bw_cl = classification_model['final model'].predict(input) # 1D
    bw_cl = bw_cl.reshape(n_row, n_col) # 2D
    bw_cl = bw_cl.astype(bool)
    bw_cl[bw_crop==False] = False

    # Make a pseudo image of map
    r[bw_cl == True] = 1
    g[bw_cl == True] = 0
    b[bw_cl == True] = 0
    pseu_cl = np.concatenate((r.reshape(n_row, n_col, 1),
                              g.reshape(n_row, n_col, 1),
                              b.reshape(n_row, n_col, 1)),
                             axis=2)

    pseu_cl = exposure.adjust_gamma(pseu_cl, gamma)

    # Show result
    if flag_figure:
        fig, ax = plt.subplots(1, 2)
        fig.suptitle('Classification map')
        ax[0].imshow(bw_cl, cmap='gray')
        ax[0].set_title('BW of classification')
        ax[1].imshow(pseu_cl)
        ax[1].set_title('Pseudo RGB R=' + str(wave_r) + ' G=' + str(wave_g) + ' B=' + str(wave_b))
        plt.show()

    print('All done!')

    return {'bw_crop': bw_crop,
            'pseu_crop': pseu_crop,
            'bw_classification': bw_cl,
            'pseu_classification': pseu_cl}


######
# Binary classification report 
######
# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,      # This is the area under the PR curve (AP)
    precision_recall_curve,       # For plotting PR curve across thresholds
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)

def binary_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label=1,
    print_report: bool = True,
) -> Dict[str, Any]:
    """
    Compute a set of binary classification metrics and optionally plot ROC/PR curves.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (e.g., 0/1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (e.g., 0/1).
   
    pos_label : int or str, default=1
        The label of the positive class.
    
    Returns
    -------
    results : dict
        Dictionary containing the computed metrics and curve data (if available).
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
   
    # --- Basic scalar metrics (no averaging needed for binary) ---
    cm = confusion_matrix(y_true, y_pred, labels=[0, pos_label])
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    # --- Optional printing ---
    if print_report:
        print("\n=== Binary Classification Report ===")
        print(f"Accuracy:      {acc:.4f}")
        print(f"Precision:     {precision:.4f}")
        print(f"Recall:        {recall:.4f}")
        print(f"F1-score:      {f1:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print("\nConfusion Matrix (rows: true, cols: pred):")
        print(cm)

        # A succinct per-class report (precision/recall/F1/support)
        print("\nPer-class metrics:")
        print(classification_report(y_true, y_pred, digits=4))

    # --- Return everything in a dict for programmatic access ---
    results = {
        'confusion_matrix': cm,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "pos_label": pos_label,
    }
    return results


######
# pr_curve_report
######
# from typing import Dict, Any, Optional
# import numpy as np
# from sklearn.metrics import (
# precision_recall_curve,
# roc_curve,
# roc_auc_score,
# average_precision_score,
# accuracy_score,
# precision_score,
# recall_score,
# f1_score,
# cohen_kappa_score,
# confusion_matrix,
# )

def pr_curve_report(
    y_true,
    y_score,
    pos_label=1,
    strategy: str = "f1",
    beta: float = 1.0,
    target_precision: Optional[float] = None,
    target_recall: Optional[float] = None,
    return_curves: bool = True,
) -> Dict[str, Any]:
    """
    Select the best threshold from the Precision-Recall curve and compute evaluation metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels. Can be {0,1}, booleans, or any labels; `pos_label` indicates the positive class.
    y_score : array-like of shape (n_samples,)
        Predicted scores for the positive class (e.g., probabilities or decision function values).
    pos_label : Any, default=1
        The label of the positive class in `y_true`.
    strategy : {"f1", "f_beta", "target_precision", "target_recall"}, default="f1"
        Threshold selection strategy:
          - "f1": maximize F1
          - "f_beta": maximize Fβ (β>1 emphasizes recall; β<1 emphasizes precision)
          - "target_precision": choose highest threshold that achieves precision ≥ target_precision
          - "target_recall": choose highest threshold that achieves recall ≥ target_recall
    beta : float, default=1.0
        β parameter for Fβ when strategy="f_beta".
    target_precision : float in (0,1], optional
        Required when strategy="target_precision".
    target_recall : float in (0,1], optional
        Required when strategy="target_recall".
    return_curves : bool, default=True
        If True, include PR and ROC curve arrays in the returned dict.

    Returns
    -------
    results : dict
        {
            'confusion_matrix': np.ndarray([[TN, FP], [FN, TP]]),
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float,
            "kappa": float,
            "roc_auc": float,
            "pr_auc": float,         # Average Precision (AP)
            "roc_curve": dict|None,  # {'fpr', 'tpr', 'thresholds'}
            "pr_curve": dict|None,   # {'precision', 'recall', 'thresholds', 'best_threshold'}
            "pos_label": pos_label,
        }
    """

    # --- Validate and binarize y_true with respect to pos_label ---
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains NaN or infinite values.")
    y_bin = (y_true == pos_label).astype(int)

    # --- Precision-Recall curve ---
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_bin, y_score)
    # thresholds[i] corresponds to precision[i+1], recall[i+1]
    p = pr_precision[1:]
    r = pr_recall[1:]
    th = pr_thresholds

    if len(th) == 0:
        # Degenerate case: model outputs a constant score -> fallback to threshold=0.5
        best_threshold = 0.5
    else:
        # --- Choose threshold according to strategy ---
        if strategy == "f1":
            denom = (p + r) + 1e-10
            f1_vals = np.where(denom > 0, 2 * p * r / denom, 0.0)
            best_idx = int(np.argmax(f1_vals))
            best_threshold = th[best_idx]

        elif strategy == "f_beta":
            if beta <= 0:
                raise ValueError("beta must be > 0 for F_beta.")
            denom = (beta**2 * p + r) + 1e-10
            f_beta_vals = np.where(denom > 0, (1 + beta**2) * p * r / denom, 0.0)
            best_idx = int(np.argmax(f_beta_vals))
            best_threshold = th[best_idx]

        elif strategy == "target_precision":
            if target_precision is None:
                raise ValueError("target_precision must be provided when strategy='target_precision'.")
            # indices where precision >= target; pick the one with the largest threshold (more selective)
            idx = np.where(p >= target_precision)[0]
            if len(idx) > 0:
                best_idx = int(idx[-1])
                best_threshold = th[best_idx]
            else:
                # fallback to best F1 if target cannot be met
                denom = (p + r) + 1e-10
                f1_vals = np.where(denom > 0, 2 * p * r / denom, 0.0)
                best_idx = int(np.argmax(f1_vals))
                best_threshold = th[best_idx]

        elif strategy == "target_recall":
            if target_recall is None:
                raise ValueError("target_recall must be provided when strategy='target_recall'.")
            # r is (mostly) non-increasing with threshold; choose highest threshold with recall >= target
            idx = np.where(r >= target_recall)[0]
            if len(idx) > 0:
                best_idx = int(idx[-1])
                best_threshold = th[best_idx]
            else:
                # fallback to best F1 if target cannot be met
                denom = (p + r) + 1e-10
                f1_vals = np.where(denom > 0, 2 * p * r / denom, 0.0)
                best_idx = int(np.argmax(f1_vals))
                best_threshold = th[best_idx]
        else:
            raise ValueError("Unknown strategy. Use 'f1', 'f_beta', 'target_precision', or 'target_recall'.")

    # --- Predict with the selected threshold ---
    y_pred = (y_score >= best_threshold).astype(int)

    # --- Scalar metrics ---
    acc = accuracy_score(y_bin, y_pred)
    prec = precision_score(y_bin, y_pred, zero_division=0)
    rec = recall_score(y_bin, y_pred, zero_division=0)
    f1 = f1_score(y_bin, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_bin, y_pred)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_bin, y_pred, labels=[0, 1])  # [[TN, FP], [FN, TP]]

    # --- ROC / PR AUC ---
    # roc_auc = roc_auc_score(y_bin, y_score)
    pr_auc = average_precision_score(y_bin, y_score)  # AP (area under PR via average precision)

    # --- Curves (optional) ---
    # roc_curve_data = None
    pr_curve_report = None
    if return_curves:
        fpr, tpr, roc_th = roc_curve(y_bin, y_score)
        # roc_curve_data = {"fpr": fpr, "tpr": tpr, "thresholds": roc_th}
        pr_curve_report = {
            "precision": pr_precision,
            "recall": pr_recall,
            "thresholds": pr_thresholds,
            'pr_auc': pr_auc,
            "best_threshold": best_threshold            
        }

    bi_classification_best_th_report = {
        'confusion_matrix': cm,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "kappa": kappa,
        "pos_label": pos_label        
    }
    return bi_classification_best_th_report, pr_curve_report

# ----------------------------
# Example usage (remove/modify)
# ----------------------------
if __name__ == "__main__":
    rng = np.random.RandomState(42)
    n = 300

    # Synthetic ground truth with imbalance (positive rate ~ 0.3)
    y_true = (rng.rand(n) < 0.3).astype(int)

    # Synthetic predicted probabilities (noisy but correlated with y_true)
    y_score = 0.2 * rng.rand(n) + 0.6 * y_true + 0.1 * rng.randn(n)
    y_score = 1 / (1 + np.exp(-y_score))  # squash to (0,1)

    # Choose a threshold (e.g., 0.5) to get hard predictions
    threshold = 0.5
    y_pred = (y_score >= threshold).astype(int)

    # Compute and plot
    # _ = binary_classification_report(
    #     y_true=y_true,
    #     y_pred=y_pred,
    #     y_score=y_score,
    #     pos_label=1,
    #     print_report=True,
    #     # plot_curves=True,
    # )

    sdt_rep, pr_curve_rep = pr_curve_report(
                    y_true=y_true,
                    y_score=y_score,
                    pos_label=1,
                    strategy="f1",
                    beta=1.0,
                    target_precision=None,
                    target_recall=None,
                    return_curves=True
                )

    print('Stand bi-classification report:\n', sdt_rep)
    print('\npr-curve report: \n', pr_curve_rep)

# ########################################################################################################################
# # Demo of make_class_map
# ########################################################################################################################
# if __name__ == "__main__":
#     import sys
#     import numpy as np
#     sys.path.append('/media/huajian/Files/python_projects/appf_toolbox_project')
#     from appf_toolbox.hyper_processing import envi_funs
#
#     # Input parameters
#     data_path = '/media/huajian/Files/Data/crown_rot_0590/wiw_top_view_examples'
#     data_name = 'vnir_100_140_9064_2021-08-10_01-08-07'
#
#     seg_model_path = '/media/huajian/Files/python_projects/p12_green_segmentation/green_seg_model_20220201'
#     seg_model_name = 'record_OneClassSVM_vnir_hh.sav'
#
#     classification_model_path = '/media/huajian/Files/python_projects/crown_rot_0590/crown_rot_0590_processed_data/' \
#                         'crown_rot_0590_top_view/2c_models'
#     classification_model_name = 'model_svm_2c_vnir_hyp-hue_top_view_2021-08-03.sav'
#
#     # Read the data
#     raw_data, meta_plant = envi_funs.read_hyper_data(data_path, data_name)
#
#     wavelengths = np.zeros((meta_plant.metadata['Wavelength'].__len__(),))
#     for i in range(wavelengths.size):
#         wavelengths[i] = float(meta_plant.metadata['Wavelength'][i])
#
#     # Calibrate the data
#     hypcube = envi_funs.calibrate_hyper_data(raw_data['white'], raw_data['dark'], raw_data['plant'],
#                                              trim_rate_t_w=0.1, trim_rate_b_w=0.95)
#
#     # Make map
#     map = make_class_map(hypcube,
#                          wavelengths,
#                          seg_model_path,
#                          seg_model_name,
#                          classification_model_path,
#                          classification_model_name,
#                          band_r=97,
#                          band_g=52,
#                          band_b=14,
#                          gamma=0.7,
#                          flag_remove_noise=True,
#                          flag_figure=True)

