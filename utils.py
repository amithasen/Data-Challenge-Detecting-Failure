from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt


def model_validation(y_test, y_pred):
    conf = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print('confusion matrix')
    print(conf, '\n')
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('accuracy: {}'.format(accuracy))


def plot_model_curves(clf, test_set, y_test, decision_func=False):
    if decision_func:
        probs = clf.decision_function(test_set)
    else:
        probs = clf.predict_proba(test_set)
        probs = probs[:, 1]  # use probability for anomaly class

    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].set_title('ROC Curve - Positive Class')
    axs[0].plot([0, 1], [0, 1], linestyle='--')
    axs[0].plot(fpr, tpr, marker='.', label='ROC Curve (area = {:0.2f})'.format(roc_auc))
    axs[0].set_xlabel('false positive rate')
    axs[0].set_ylabel('true positive rate')
    axs[0].legend(loc='lower right')

    axs[1].set_title('precision-recall curve')
    axs[1].plot([0, 1], [0.5, 0.5], linestyle='--')
    axs[1].plot(recall, precision, marker='.', label='Precision/Recall Curve (area = {:0.2f})'.format(pr_auc))
    axs[1].set_xlabel('recall')
    axs[1].set_ylabel('precision')
    axs[1].legend(loc='upper right')

    plt.show()


def get_performance_measure(y_actual, y_pred):

        tp = fp = tn = fn = 0
        for i in range(len(y_pred)):

            if y_actual[i] == y_pred[i] == 1:
                tp += 1
            if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
                fp += 1
            if y_actual[i] == y_pred[i] == 0:
                tn += 1
            if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
                fn += 1

        print('True Positives ', tp)
        print('False Positives ', fp)
        print('True Negatives ', tn)
        print('False Negatives ', fn)