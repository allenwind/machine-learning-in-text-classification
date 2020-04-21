from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# 评估指标

__all__ = ["plot_roc", "plot_rocs", "plot_pc", "plot_pcs",
           "plot_label_distribution"]

def plot_f1_curve(y_true, y_pred, label, show=True):
    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_pred)
    f1 = 2 * precision * recall / (precision + recall)
    plt.plot(threshold, f1[:-1], label="{} f1-curve".format(label))
    i = np.argmax(f1)
    plt.plot(threshold[i], f1[i], "o", label="max f1")

    plt.xlabel("threshold")
    plt.ylabel("f1")
    plt.title("f1 curve")
    plt.legend(loc="lower right")
    if show:
        plt.show()

def plot_roc(y_true, y_predict, label, show=True, baseline=False, save=False):
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="random-chance", alpha=0.6)
    plt.plot(fpr, tpr, label="{} auc={:.3f}".format(label, auc))

    if baseline:
        plt.axvline(x=0.005, color="red", label="0.5% FPR", alpha=0.8)
        plt.axvline(x=0.01, color="red", label="1% FPR", alpha=0.8)
        plt.axhline(y=0.9, color="red", label="90% TPR", alpha=0.8)

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("roc curve")
    plt.legend(loc="lower right")
    if save:
        plt.savefit("-".join(labels), quality=95)
    if show:
        plt.show()

def plot_rocs(ys_true, ys_predict, labels, show=True, baseline=False, save=False):
    for y_true, y_predict, label in zip(ys_true, ys_predict, labels):
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_predict, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label="{} auc={:.3f}".format(label, auc))

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="random-chance", alpha=0.6)
    if baseline:
        plt.axvline(x=0.005, color="red", label="0.5% FPR", alpha=0.8)
        plt.axvline(x=0.01, color="red", label="1% FPR", alpha=0.8)
        plt.axhline(y=0.9, color="red", label="90% TPR", alpha=0.8)

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("roc curve")
    plt.legend(loc="lower right")
    if save:
        plt.savefit("-".join(labels), quality=95)
    if show:
        plt.show()

def plot_pc(y_true, y_predict, label, show=True):
    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_predict)
    plt.plot(recall, precision, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="balance", alpha=0.6)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.title("precision recall curve")
    if show:
        plt.show()

def plot_pcs(ys_true, ys_predict, labels, show=True):
    for y_true, y_predict, label in zip(ys_true, ys_predict, labels):
        precision, recall, threshold = metrics.precision_recall_curve(y_true, y_predict)
        plt.plot(recall, precision, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="balance", alpha=0.6)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.title("precision recall curve")
    if show:
        plt.show()

def plot_label_distribution(labels):
    x = labels[:, 0]
    y = labels[:, 1]
    
    bins = len(labels) // 10
    freqs, locs = np.histogram(x, bins=bins)
    plt.plot(locs[1:], freqs)   
    freqs, locs = np.histogram(y, bins=bins)
    plt.plot(locs[1:], freqs)
    plt.show()
