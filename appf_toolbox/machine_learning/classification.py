########################################################################################################################
def plot_samples_with_colourbar(samples, labels, wavelengths=[], input_type='Input data values', title='Title'):
    """
    Plot the samples (reflectance values) with a colour bar which is defined by the values of the labels.

    :param samples: input data array; usually reflectance values
    :param labels: the values of the labels (the parameter need to be measured)
    :param x_axis_value: the x_axis_value in the plot; default is [] which will lead to x axis values of 1, 2, 3 ......
    :param input_type: reflectance, pca, etc
    :param title: title for plot
    :return: return 0 if no errors
    """
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    import numpy as np


    # n_lines = 5
    # x = np.linspace(0, 10, 100)
    # y = np.sin(x[:, None] + np.pi * np.linspace(0, 1, n_lines))
    # c = np.arange(1., n_lines + 1)
    #
    # cmap = plt.get_cmap("jet", len(c))
    # norm = colors.BoundaryNorm(np.arange(len(c) + 1) + 0.5, len(c))
    # sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1
    #
    # fig, ax = plt.subplots(dpi=100)
    # for i, yi in enumerate(y.T):
    #     ax.plot(x, yi, c=cmap(i))
    # fig.colorbar(sm, ticks=c)
    # plt.show()

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


def repeadted_kfold_cv(input, label, n_splits, n_repeats, tune_model, karg, random_state=0,
                       flag_save=False,
                       file_name_save='cv_record'):
    """
    Perform repeated k-folds cross validation of classification. V1.0 only tested for binary classification.

    :param input: Input data in the format of 2D numpy array.
    :param label: The ground-trued labels. 1D numpy array in int.
    :param n_splits: The number of splits for cross validation
    :param n_repeats: The number of repeat for cross validation.
    :param tune_model: The function for tuning the models.
    :param karg: Key words arguments for tune_model()
    :param random_state: Random state for cross-validation. Default is 0
    :param flag_save: Flag to save the record. If set to True, it will save the record as a .save file in the present
           working directory. Default is False
    :param file_name_save: The file name to save the record. Default is 'cv_record'.
    :return: The record of repeated cross validation.

     Version 1.0 Date: Aug 25, 2021 Tested for binary classification.
     Author: Huajian Liu huajian.liu@adelaide.edu.au
    """

    from sklearn.model_selection import RepeatedKFold
    import sklearn.metrics as met
    import numpy as np
    import joblib
    from datetime import datetime

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    # For record the result of each cross validation
    record_each_cv = {'confusion_matrix': [],
                      'recall': [],
                      'precision': [],
                      'f1': [],
                      'accuracy': [],
                      'num_pos_neg': []}

    # Performing cross validation
    count_cv = 1
    for train_ind, val_index in rkf.split(input):
        print('')
        print('==========cross validation ' + str(count_cv) + '==========')

        # Train
        tuned_model = tune_model(input[train_ind], label[train_ind], **karg)
        print('Tuned model:')
        print(tuned_model)

        # Prediction
        output_val = tuned_model.predict(input[val_index])

        # Confusion matrix
        conf_max = met.confusion_matrix(label[val_index], output_val)
        print('confusion matrix:')
        print(conf_max)
        record_each_cv['confusion_matrix'].append(conf_max)

        # Recall
        recall = met.recall_score(label[val_index], output_val)
        print('Recall:', recall)
        record_each_cv['recall'].append(recall)

        # Precision
        precision = met.precision_score(label[val_index], output_val)
        print('Precision:', precision)
        record_each_cv['precision'].append(precision)

        # f1 score
        f1 = met.f1_score(label[val_index], output_val)
        print('f1 score', f1)
        record_each_cv['f1'].append(f1)

        # Accuracy
        accuracy = met.accuracy_score(label[val_index], output_val)
        print('Accuracy', accuracy)
        record_each_cv['accuracy'].append(accuracy)

        # Number of positive and negative samples in validation
        num_pos = np.sum(label[val_index])
        num_neg = label[val_index].shape[0] - num_pos
        print('The number of positive and negative samples ', (num_pos, num_neg))
        record_each_cv['num_pos_neg'].append((num_pos, num_neg))

        count_cv += 1

    print()
    print('Summyar of ' + str(n_splits) + '-fold cross validation with ' + str(n_repeats) + ' repeats')

    # Average confusion matrix
    ave_con_mat = record_each_cv['confusion_matrix']
    ave_con_mat = np.asarray(ave_con_mat)
    ave_con_mat = np.mean(ave_con_mat, axis=0)
    ave_con_mat = np.round(ave_con_mat)
    print('Average confusion matrix: ')
    print(ave_con_mat)

    # Average recall
    ave_recall = record_each_cv['recall']
    ave_recall = np.asarray(ave_recall)
    ave_recall = np.mean(ave_recall, axis=0)
    print('Average recall: ', ave_recall)

    # Average precision
    ave_precision = record_each_cv['precision']
    ave_precision = np.asarray(ave_precision)
    ave_precision = np.mean(ave_precision, axis=0)
    print('Average precision: ', ave_precision)

    # Average f1 score
    ave_f1 = record_each_cv['f1']
    ave_f1 = np.asarray(ave_f1)
    ave_f1 = np.mean(ave_f1, axis=0)
    print('Average f1 score: ', ave_f1)

    # Average accuracy
    ave_accuracy = record_each_cv['accuracy']
    ave_accuracy = np.asarray(ave_accuracy)
    ave_accuracy = np.mean(ave_accuracy, axis=0)
    print('Average accuracy: ', ave_accuracy)

    # The number of samples of each class; total samples
    classes, counts = np.unique(label, return_counts=True)
    total_samples = label.shape[0]
    print('Total samples: ', total_samples)
    print('Classes:', classes)
    print('Count in each classes: ', counts)

    # Train a final model using all of the data
    final_model = tune_model(input, label, **karg)

    record = {'record of each cv': record_each_cv,
              'average confusion matrix': ave_con_mat,
              'average recall': ave_recall,
              'average f1': ave_f1,
              'average accuracy': ave_accuracy,
              'total samples': total_samples,
              'classes: ': classes,
              'count in each classes': counts,
              'final model': final_model}

    if flag_save:
        file_name_save = file_name_save + '_' + datetime.now().strftime('%y-%m-%d-%H-%M-%S') + '.sav'
        joblib.dump(record, file_name_save)

    return record


def tune_svm_classification(input,
                            label,
                            svm_kernel='rbf',
                            svm_c_range=[1, 100],
                            svm_gamma_range=[1, 50],
                            svm_tol=1e-3,
                            opt_num_iter_cv=5,
                            opt_num_fold_cv=5,
                            opt_num_evals=5):
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
    :return: A tuned SVM model for binary classification.

    Author: Huajina Liu email: huajina.liu@adelaide.edu.au
    Version: 1.0 Date: August 20, 2021
    """
    from sklearn.svm import SVC
    import optunity.metrics

    # Search the optimal parameters
    @optunity.cross_validated(x=input, y=label, num_iter=opt_num_iter_cv, num_folds=opt_num_fold_cv)
    def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
        model = SVC(kernel=svm_kernel, C=C, gamma=gamma, tol=svm_tol)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.error_rate(y_test, predictions) # error_rate = 1.0 - accuracy(y, yhat)

    optimal_pars, _, _ = optunity.minimize(tune_cv, num_evals=opt_num_evals, C=svm_c_range, gamma=svm_gamma_range)

    # Train a model using the optimal parameters
    tuned_model = SVC(**optimal_pars).fit(input, label)
    return tuned_model


