import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def create_labels(n=100, bad_rate=0.5):
    # generate an array of binary samples (0 or 1), distributed by bad_rate
    X = np.random.binomial(1, bad_rate, n)
    return X


def create_random_predictions(labels: np.ndarray):
    # generate an array of random predictions floating between 0 and 1,
    # uniformly distributed
    predictions = np.random.uniform(0, 1, len(labels))
    return predictions


def create_perfect_predictions(labels: np.ndarray):
    # generate an array of predictions that perfectly match the labels
    predictions = np.array(labels).astype(float)
    return predictions


def create_noisy_predictions(labels: np.ndarray, noise_level=0.1):
    # predict the labels with a small amount of noise
    predictions = np.array(labels).astype(float) + np.random.normal(noise_level, size=len(labels))
    # cap predictions between 0 and 1
    predictions[predictions < 0] = 0.0
    predictions[predictions > 1] = 1.0
    return predictions


def calculate_roc_auc(labels: np.ndarray, predictions: np.ndarray):
    # calculate the area under the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def calculate_pr_auc(labels: np.ndarray, predictions: np.ndarray):
    # calculate the area under the PR curve
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    pr_auc = auc(recall, precision)
    return pr_auc


def plot_roc_curve(labels: np.ndarray, predictions: np.ndarray):
    # plot the ROC curve
    fig, ax = plt.subplots()
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="best")
    # plt.show()
    return fig


def plot_pr_curve(labels: np.ndarray, predictions: np.ndarray):
    # plot the PR curve
    fig, ax = plt.subplots()
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.axhline(labels.mean(), color='red', linestyle=':',
                label='Mean positive rate')
    plt.legend(loc="best")
    # plt.show()
    return fig


if __name__ == '__main__':
    y = create_labels(n=10000, bad_rate=0.1)
    y_score = create_random_predictions(y)
    # y_score = create_perfect_predictions(y)
    y_score = create_noisy_predictions(y, noise_level=0.2)
    plot_roc_curve(y, y_score)
    plot_pr_curve(y, y_score)
    plt.show()