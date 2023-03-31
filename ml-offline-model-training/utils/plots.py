import matplotlib.pyplot as plt


def plot_roc_curve(roc_curve_info, file_path='roc_curve.png'):
    plt.figure(figsize=(10, 8))
    for i in range(len(roc_curve_info)):
        label = f'Fold {i} validation ROC curve' if len(roc_curve_info) > 1 else 'Test ROC curve'
        plt.plot(
            roc_curve_info[i]['fpr'],
            roc_curve_info[i]['tpr'],
            label=label)
    plt.legend(loc='lower right')
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(file_path)
