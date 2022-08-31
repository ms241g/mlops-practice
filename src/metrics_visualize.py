import time
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from matplotlib import pyplot as plt
import numpy as np
import math
import seaborn as sns

def ml_model(classifier, classifier_name, **kwargs):
    # Fit model
    if kwargs['x_train'] is not None:
        model = classifier.fit(kwargs['x_train'], kwargs['y_train'])
        y_pred_train= model.predict(kwargs['x_train'])
        accuracy = accuracy_score(kwargs['y_train'], y_pred_train)
        roc_auc = roc_auc_score(kwargs['y_train'], y_pred_train, average='weighted')
        cm = confusion_matrix(kwargs['y_train'], y_pred_train)

        print("Accuracy of",classifier_name,": {:.2f}".format(accuracy))
        print("ROC AUC Score of", classifier_name,": {:.2f}".format(roc_auc))
        print("Confusion Matrix of", classifier_name, cm)

        #plt.figure(figsize=(5,5))
        #sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues');
        #plt.ylabel('Actual label');
        #plt.xlabel('Predicted label');
        #title = 'AUC-ROC Score: {:.2f}'.format(roc_auc)
        #plt.title(title)
        #plt.show()
    
    if kwargs['x_valid'] is not None:
        y_pred_valid = model.predict(kwargs['x_valid'])
        print('*****************************************************')
        print('Validation Set Performance:')
        print('*****************************************************')
        #evaluatation_metrics(kwargs['y_valid'], y_pred_valid, classifier_name)
        accuracy = accuracy_score(kwargs['y_valid'], y_pred_valid)
        roc_auc = roc_auc_score(kwargs['y_valid'], y_pred_valid, average='weighted')
        cm = confusion_matrix(kwargs['y_valid'], y_pred_valid)

        print("Accuracy of",classifier_name,": {:.2f}".format(accuracy))
        print("ROC AUC Score of", classifier_name,": {:.2f}".format(roc_auc))
        print("Confusion Matrix of", classifier_name, cm)

        #plt.figure(figsize=(5,5))
        #sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues');
        #plt.ylabel('Actual label');
        #plt.xlabel('Predicted label');
        #title = 'AUC-ROC Score: {:.2f}'.format(roc_auc)
        #plt.title(title)
        #plt.show()
    
    if kwargs['x_test'] is not None:
        start = time.time()
        y_pred_test= classifier.predict(kwargs['x_test'])
        end = time.time()
        print('*****************************************************')
        print('Test Set Performance:')
        print('*****************************************************')
        print('Model Time Complexity on Test Data: {:.3f} milli seconds'.format((end - start) * 1000))
        #evaluatation_metrics(kwargs['y_test'], y_pred_test, classifier_name)
        accuracy = accuracy_score(kwargs['y_test'], y_pred_test)
        roc_auc = roc_auc_score(kwargs['y_test'], y_pred_test, average='weighted')
        cm = confusion_matrix(kwargs['y_test'], y_pred_test)

        print("Accuracy of",classifier_name,": {:.2f}".format(accuracy))
        print("ROC AUC Score of", classifier_name,": {:.2f}".format(roc_auc))
        print("Confusion Matrix of", classifier_name,": \n")

        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        title = 'AUC-ROC Score: {:.2f}'.format(roc_auc)
        plt.title(title)
        plt.savefig("learning_curve_test.jpg")


def generate_learning_curves(model, model_name, X, y, xlim = None, ylim=None, 
                         epochs =None, figsize = (20,5)):
    cross_valid = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    
    train_sizes=np.linspace(.1, 1.0, 5)
    train_sizes, train_scores, test_scores, training_time, _ = learning_curve(model, X, y, cv=cross_valid, 
                                                                           n_jobs=epochs, train_sizes=train_sizes,
                                                                           return_times=True)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    training_time_mean = np.mean(training_time, axis=1)
    training_time_std = np.std(training_time, axis=1)


    plt.title(model_name)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig("learning_curve.jpg")