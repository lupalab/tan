import numpy as np


def anomally_detection(log_llk, label):
    idx = np.argsort(log_llk)
    total_outliers = np.sum(label)

    rcall = [0]*len(log_llk)
    pcision = [0]*len(log_llk)
    num_outliers = 0.0
    avg_precision = 0.0
    for i in range(len(log_llk)):
        num_outliers += label[idx[i]]
        rcall[i] = num_outliers / total_outliers
        pcision[i] = num_outliers/(i+1.0)
        if i == 0:
            delta_r = rcall[i]
        else:
            delta_r = rcall[i]- rcall[i-1]
        avg_precision += delta_r * pcision[i]
    dic = {
        'recall': rcall, 'precision': pcision,
        'avg_precision': avg_precision}
    return dic
