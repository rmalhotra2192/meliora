import pandas as pd
from sklearn import metrics


def bayesian_error_rate(default_flag, prob_default):
    """
    BER is the proportion of the whole sample that is misclassified
    when the rating system is in optimal use. For a perfect rating model,
    the BER has a value of zero. A model's BER depends on the probability
    of default. The lower the BER, and the lower the classification error,
    the better the model.

    The Bayesian error rate specifies the minimum probability of error if
    the rating system or score function under consideration is used for a
    yes/no decision whether a borrower will default or not. The error can
    be estimated parametrically, e.g. assuming normal score distributions,
    or non-parametrically, for instance with kernel density estimation methods.

    If parametric estimation is applied, the distributional assumptions have
    to be carefully checked. Non-parametric estimation will be critical if
    sample sizes are small. In its general form, the error rate depends on
    the total portfolio probability of default. As a consequence, in many
    cases its magnitude is influenced much more by the probability of
    erroneously identifying a non-defaulter as a defaulter than by the
    probability of not detecting a defaulter.

    In practice, therefore, the error rate is often applied
    with a fictitious 50% probability of default. In this case, the error
    rate is equivalent to the Kolmogorov-Smirnov statistic and to the Pietra index.

    Parameters
    ----------
    default_flag : pandas series
        Boolean flag indicating whether the borrower has actually defaulted
    prob_default : pandas series
        Predicted default probability, as returned by a classifier.

    Returns
    -------
    score : float
        Return Bayesian Error Rate

    """

    frame = {'default_flag': default_flag,
             'prob_default': prob_default
             }

    df = pd.DataFrame(frame)

    fpr, tpr, thresholds = metrics.roc_curve(
        df['default_flag'], df['prob_default'])
    roc_curve_df = pd.DataFrame({'c': thresholds,
                                'hit_rate': tpr,
                                 'false_alarm_rate': fpr})

    p_d = df.default_flag.sum()/len(df)

    roc_curve_df['ber'] = p_d*(1 - roc_curve_df.hit_rate) + \
        (1 - p_d) * roc_curve_df.false_alarm_rate

    return round(min(roc_curve_df['ber']), 3)
