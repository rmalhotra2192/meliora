from scipy import stats


def ks(rvs, cdf, args=(), N=20, alternative='two-sided', mode='auto'):
    """
    KS is the maximum distance between two population distributions. 

    This statistic helps discriminate default accounts from non-default 
    accounts. It is also used to determine the best cutoff in application 
    scoring. The best cutoff maximizes KS, which becomes the best 
    differentiator between the two populations. The KS value can range 
    between 0 and 1, where 1 implies that the model is perfectly accurate 
    in predicting default accounts or separating the two populations. 
    A higher KS denotes a better model.

    Performs the (one-sample or two-sample) Kolmogorov-Smirnov test for
    goodness of fit.
    The one-sample test compares the underlying distribution F(x) of a sample
    against a given distribution G(x). The two-sample test compares the
    underlying distributions of two independent samples. Both tests are valid
    only for continuous distributions.

    Parameters
    ----------
    rvs : str, array_like, or callable
        If an array, it should be a 1-D array of observations of random
        variables.
    cdf : str, array_like or callable
        If array_like, it should be a 1-D array of observations of random
        variables, and the two-sample test is performed
        (and rvs must be array_like).

    N : int, optional
        Sample size if 'rvs' is string or callable.  Default is 20.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the null and alternative hypotheses. Default is 'two-sided'.


    Returns
    -------
    statistic : float
                KS test statistic, either D, D+ or D-.
    pvalue :  float
                One-tailed or two-tailed p-value.


    Examples
    --------
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-15, 15, 9)
    >>> stats.kstest(x, 'norm')
     KstestResult(statistic=0.444356027159..., pvalue=0.038850140086...)

    """

    return stats.kstest(rvs, cdf, args=(), N=20, alternative='two-sided', mode='auto')
