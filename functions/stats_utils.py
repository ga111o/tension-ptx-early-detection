import numpy as np
from scipy import stats

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = positive_examples.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n

    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    return aucs, delongcov

def compute_delong_pvalue(y_true, y_pred1, y_pred2):
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    order = np.argsort(y_true)[::-1]
    y_true_sorted = y_true[order]
    
    if y_true_sorted[0] != 1:
        pos_idx = y_true == 1
        neg_idx = y_true == 0
        y_true_sorted = np.concatenate([y_true[pos_idx], y_true[neg_idx]])
        y_pred1 = np.concatenate([y_pred1[pos_idx], y_pred1[neg_idx]])
        y_pred2 = np.concatenate([y_pred2[pos_idx], y_pred2[neg_idx]])
    else:
        y_pred1 = y_pred1[order]
        y_pred2 = y_pred2[order]

    label_1_count = int(y_true.sum())
    
    predictions_sorted_transposed = np.vstack((y_pred1, y_pred2))
    
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    
    # z-score
    l = np.array([1, -1])
    # Variance of the difference
    sigma_diff = np.dot(np.dot(l, delongcov), l.T)
    z = (aucs[0] - aucs[1]) / np.sqrt(sigma_diff)
    
    # Two-sided p-value
    p_value = 2 * stats.norm.sf(np.abs(z))
    return p_value, aucs[0], aucs[1]

def perform_paired_ttest(scores1, scores2):
    """
    Perform paired t-test between two sets of scores (e.g., from cross-validation folds).
    """
    t_stat, p_val = stats.ttest_rel(scores1, scores2)
    return t_stat, p_val
