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
        If a callable, it should be a function to generate random variables;
        it is required to have a keyword argument 'size'.
        If a string, it should be the name of a distribution in 'scipy.stats',
        which will be used to generate random variables.
    cdf : str, array_like or callable
        If array_like, it should be a 1-D array of observations of random
        variables, and the two-sample test is performed
        (and rvs must be array_like).
        If a callable, that callable is used to calculate the cdf.
        If a string, it should be the name of a distribution in 'scipy.stats',
        which will be used as the cdf function.
    args : tuple, sequence, optional
            Distribution parameters, used if 'rvs' or 'cdf' are strings or
            callables.

    N : int, optional
        Sample size if 'rvs' is string or callable.  Default is 20.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the null and alternative hypotheses. Default is 'two-sided'.
        Please see explanations in the Notes below.
    mode : {'auto', 'exact', 'approx', 'asymp'}, optional
        Defines the distribution used for calculating the p-value.
        The following options are available (default is 'auto'):
          * 'auto' : selects one of the other options.
          * 'exact' : uses the exact distribution of test statistic.
          * 'approx' : approximates the two-sided probability with twice the
            one-sided probability
          * 'asymp': uses asymptotic distribution of test statistic

    Returns
    -------
    statistic : float
                KS test statistic, either D, D+ or D-.
    pvalue :  float
                One-tailed or two-tailed p-value.

    See Also
    --------
    ks_2samp


    Notes
    -----
    There are three options for the null and corresponding alternative
    hypothesis that can be selected using the `alternative` parameter.

    - 'two-sided': The null hypothesis is that the two distributions are
      identical, F(x)=G(x) for all x; the alternative is that they are not
      identical.
    - 'less': The null hypothesis is that F(x) >= G(x) for all x; the
      alternative is that F(x) < G(x) for at least one x.
    - 'greater': The null hypothesis is that F(x) <= G(x) for all x; the
      alternative is that F(x) > G(x) for at least one x.
    Note that the alternative hypotheses describe the *CDFs* of the
    underlying distributions, not the observed values. For example,
    suppose x1 ~ F and x2 ~ G. If F(x) > G(x) for all x, the values in
    x1 tend to be less than those in x2.

    Examples
    --------
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-15, 15, 9)
    >>> stats.kstest(x, 'norm')
    KstestResult(statistic=0.444356027159..., pvalue=0.038850140086...)

    >>> stats.kstest(stats.norm.rvs(size=100, random_state=rng), stats.norm.cdf)
    KstestResult(statistic=0.165471391799..., pvalue=0.007331283245...)

    >>> x = stats.norm.rvs(loc=0.2, size=100, random_state=rng)
    >>> stats.kstest(x, 'norm', alternative='less')
    KstestResult(statistic=0.1002033514..., pvalue=0.1255446444...)

    >>> stats.kstest(x, 'norm', alternative='greater')
    KstestResult(statistic=0.018749806388..., pvalue=0.920581859791...)

    >>> stats.kstest(x, 'norm')
    KstestResult(statistic=0.100203351482..., pvalue=0.250616879765...)

    >>> stats.kstest(stats.t.rvs(100, size=100, random_state=rng), 'norm')
    KstestResult(statistic=0.064273776544..., pvalue=0.778737758305...)

    >>> stats.kstest(stats.t.rvs(3, size=100, random_state=rng), 'norm')
    KstestResult(statistic=0.128678487493..., pvalue=0.066569081515...)

    """

    return stats.kstest(rvs, cdf, args=(), N=20, alternative='two-sided', mode='auto')
