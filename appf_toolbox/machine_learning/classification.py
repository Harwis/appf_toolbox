from sklearn.svm import SVC
# from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix
import optunity.metrics

def bi_svm_rbf(sample_train, label_train, sample_test, lable_test, C_range, gamma_range,
               num_iter_inner_cv, num_fold_inner_cv, num_evals_inner):

    # sklearn.svm.SVC parameters:
    # C: Penalty parameter C of the error term. Default = 1.0
    # gamma: Kernel coefficient for 'rbf', poly and sigmoid. float, optional, default ='auto'
    # tol: float, optional, default = 1e-3

    # Search the optimal parameters
    @optunity.cross_validated(x=sample_train, y=label_train, num_iter=num_iter_inner_cv, num_folds=num_fold_inner_cv)
    def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
        model = SVC(kernel='rbf', C=C, gamma=gamma)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.error_rate(y_test, predictions)

    optimal_pars, _, _ = optunity.minimize(tune_cv, num_evals=num_evals_inner, C=C_range, gamma=gamma_range)

    tuned_model = SVC(**optimal_pars).fit(sample_train, label_train)

    # svc_rbf = SVC(kernel='rbf')
    # svc_rbf.fit(sample_train, label_train)

    label_pre = tuned_model.predict(sample_test)
    return {'Parameters': optimal_pars,
            'Confusion matrix': confusion_matrix(lable_test, label_pre),
            'Classification report': classification_report(lable_test, label_pre),
            'tuned_model': tuned_model}








from mxnet import nd, autograd, gluon
import mxnet as mx
import numpy as np


def bi_nn(samples_train, labels_train, samples_test, labels_test, batch_size, epochs, learning_rate):
    """Define a nn model for binary classification and then test the model."""

    data_ctx = mx.gpu()
    model_ctx = mx.gpu()

    # Note mxnet.ndarray is similar to numpy.ndarray in some aspects. But the differences are not negligible. For instance:
    # mxnet.ndarray.NDArray.T does real data transpose to return new a copied array, instead of returning a view of the
    # input array. mxnet.ndarray.dot performs dot product between the last axis of the first input array and the first axis
    # of the second input, while numpy.dot uses the second last axis of the input array. In addition, mxnet.ndarray.NDArray
    # supports GPU computation and various neural network layers.

    # !! Lables should be reshaped,
    labels_train = np.reshape(labels_train, (labels_train.size, 1))
    labels_test = np.reshape(labels_test, (labels_test.size, 1))

    samples_train = nd.array(samples_train)
    labels_train = nd.array(labels_train)
    samples_test = nd.array(samples_test)
    labels_test = nd.array(labels_test)

    ########################################################################################################################
    # Define a nn model and trainer; loss function
    # Instantiage a data loader
    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(samples_train, labels_train),
                                       batch_size=batch_size,
                                       shuffle=True)
    test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(samples_test, labels_test),
                                      batch_size=batch_size,
                                      shuffle=True)


    # Define the model and Loss function
    # units (int) – Dimensionality of the output space.
    # https://mxnet.incubator.apache.org/api/python/gluon/nn.html#mxnet.gluon.nn.Dense
    net = gluon.nn.Dense(1, dtype='float32')
    net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

    # Instantiage an optimizer
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})

    # Define log loss
    def logistic(z):
        return 1. / (1. + nd.exp(-z))

    def log_loss(output, y):
        yhat = logistic(output)
        return - nd.nansum(y * nd.log(yhat) + (1-y) * nd.log(1-yhat))
    ########################################################################################################################


    ########################################################################################################################
    # Train
    loss_sequence = []
    # num_train_data = len(samples_train)

    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            with autograd.record():
                output = net(data)
                loss = log_loss(output, label)
                # print('loss', loss)
                loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.sum(loss).asscalar()
        # print('Epoch %s, loss: %s' % (e, cumulative_loss))
        loss_sequence.append(cumulative_loss)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(loss_sequence)
    ########################################################################################################################


    ########################################################################################################################
    # Calculating accuracy
    num_correct = 0.0
    num_test = len(samples_test)
    for i, (data, label) in enumerate(test_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        output = net(data)
        prediction = (nd.sign(output) + 1) / 2
        num_correct += nd.sum(prediction == label)
        accuracy = num_correct.asscalar()/num_test
    # print("Accuracy: %0.3f (%s/%s)" % (accuracy, num_correct.asscalar(), num_test))
    ########################################################################################################################


    return {'Model': net,
            'loss_sequence': loss_sequence,
            'Accuracy': accuracy,
            'num_test': num_test,
            'num_train': samples_train.shape[0]}



record_net_cv = []
record_svm_rbf_cv = []
for i in range(8):
    # Training and testing data. Might make cv here.
    samples_labels_test_n = samples_labels_n[(i * 10):((i + 1) * 10)]
    samples_labels_test_c = samples_labels_c[(i * 10):((i + 1) * 10)]

    ind_test = range((i * 10), ((i + 1) * 10))
    ind_train = list(range(0, 80))
    for ele in ind_test:
        ind_train.remove(ele)
    samples_labels_train_n = samples_labels_n[ind_train]
    samples_labels_train_c = samples_labels_c[ind_train]

    # samples_labels_test_n = samples_labels_n[0:10]
    # samples_labels_test_c = samples_labels_c[0:10]
    # samples_labels_train_n = samples_labels_n[10:80]
    # samples_labels_train_c = samples_labels_c[10:80]


    # Concatenate training and testing data
    samples_labels_train = np.concatenate((samples_labels_train_n, samples_labels_train_c), axis=0)
    samples_labels_test = np.concatenate((samples_labels_test_n, samples_labels_test_c), axis=0)
    np.random.shuffle(samples_labels_train)
    np.random.shuffle(samples_labels_test)

    # Separate samples and labels
    samples_train = samples_labels_train[:, 0:-1]
    labels_train = samples_labels_train[:, -1]
    samples_test = samples_labels_test[:, 0:-1]
    labels_test = samples_labels_test[:, -1]

    # --- net ----------------------------------------------------------------------------------------------------------
    # record_net = bi_nn(samples_train, labels_train, samples_test, labels_test, batch_size, epochs, learning_rate_net)
    # record_net_cv.append(record_net)
    # --- net -----------------------------------------------------------------------------------------------------------


    # SVM_rbf
    record_svm_rbf = bi_svm_rbf(samples_train, labels_train, samples_test, labels_test,
                                C_range_svm_rbf, gamma_range_svm_rbf,
                                num_iter_inner_cv_svm_rbf, num_fold_inner_cv_svm_rbf, num_evals_inner_svm_rbf)
    record_svm_rbf_cv.append(record_svm_rbf)