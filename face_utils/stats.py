import numpy as np
import scipy.stats
from scipy import stats
import scipy.stats

def round_str(coef,round_nos=2,no_zero=False):
    """Format a number to a fixed number of decimals as a zero-padded string."""
    coef = str(round(coef,round_nos))
    split = coef.split('.')
    if len(split)>1:
        if len(split[-1])==round_nos:
            pass
        elif len(split[-1])<round_nos:
            coef = coef+'0'*(round_nos-len(split[-1]))
    else: 
        coef = coef+'.'+'0'*(round_nos)
    if no_zero==True:
        return '.'+coef.split('.')[-1]
    else:
        return coef
    
def convert_ci_str(ci,round_to=3):
    """Format a (low, high) confidence interval as a '[low, high]' string."""
    return '[{}, {}]'.format(round_str(ci[0],round_to),
                             round_str(ci[1],round_to))

def stars(coef,pval,round_nos=3):
    """Return a coefficient string annotated with significance stars (*, **, ***)."""
    stars = \
    '***' if pval<0.001 else \
    '**' if pval<0.01 else \
    '*' if pval<0.05 else ''
    return round_str(coef,round_nos)+stars

def pval(pval,round_nos=3):
    """Format a p-value, collapsing very small values to '<.001'."""
    return '<.001' if pval<=.001 else round_str(pval,round_nos,True)

def odds_prob(coef,const):
    """Convert a logit intercept + coefficient into a probability."""
    return 1/(1+np.exp(-(const+coef)))

# function to calculate Cohen's d for independent samples
# Sample sizes used by the original study, kept so the historical `cat`
# argument still reproduces the published numbers.
_LEGACY_GROUP_SIZES = {"men": 5081, "women": 10800}


def cohen_effect_size(mean1, mean2, var1, var2, cat=None,
                      n1=None, n2=None, correct=False):
    """Cohen's d for two independent groups.

    Parameters
    ----------
    mean1, mean2 : float
        Group means.
    var1, var2 : float
        Group variances.
    cat : str, optional
        Legacy selector for the original study's hardcoded per-group sample
        sizes (``'men'`` -> 5081, anything else -> 10800). Used only when
        ``n1``/``n2`` are not given, for backward compatibility.
    n1, n2 : int, optional
        Explicit per-group sample sizes. If given, they override ``cat``.
        When only ``n1`` is given, ``n2`` defaults to ``n1`` (equal groups).
    correct : bool, default False
        If ``True``, use the standard pooled-SD formula
        ``sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))``.
        If ``False`` (default), reproduce the original implementation, which
        pooled the standard deviations rather than the variances. Keep the
        default to match previously published numbers; set ``True`` for a
        statistically correct pooled SD.

    Returns
    -------
    float
        The effect size ``(mean1 - mean2) / pooled_sd``.
    """
    if n1 is None:
        n1 = _LEGACY_GROUP_SIZES.get(cat, _LEGACY_GROUP_SIZES["women"])
    if n2 is None:
        n2 = n1

    if correct:
        pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    else:
        # Original behavior: pools the standard deviations (sqrt of variance).
        s1, s2 = np.sqrt(var1), np.sqrt(var2)
        pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    return (mean1 - mean2) / pooled
    
def get_confidence_interval_data(data,confidence=0.95,err=True):
    """Return the mean and confidence-interval bounds for a data sample."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    if err==True:
        return [m,h]
    else:
        return m, m-h, m+h

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2

def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    """Fast DeLong computation of AUC covariance for correlated ROC curves."""
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)

def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
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


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    """Order labels and return positive/negative counts for the DeLong test."""
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count,_ = compute_ground_truth_statistics(ground_truth,None)
    predictions_sorted_transposed = np.vstack((predictions_one,predictions_two))[:, order]
    aucs,delongcov = fastDeLong(predictions_sorted_transposed,label_1_count,None)
    return calc_pvalue(aucs,delongcov)

def get_confidence_interval(true,pred):
    """Return the confidence interval for an AUC via the DeLong variance."""
    auc,auc_cov = delong_roc_variance(true,pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - .95) / 2)
    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)
    ci[ci>1]=1
    return auc,auc_cov,ci