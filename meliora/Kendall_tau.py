from scipy import stats


def kendall_tau(x, y, variant='b'):
    """
    Calculate Kendall's tau, a correlation measure for ordinal data.

    This is a wrapper around SciPy kendalltau function.

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1;
    the hypothesis tests (their p-values) are identical. Kendall's original
    tau-a is not implemented separately because both tau-b and tau-c reduce
    to tau-a in the absence of ties.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.
    variant: {'b', 'c'}, optional
        Defines which variant of Kendall's tau is returned. Default is 'b'.


    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.

    References
    --------------
    [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.
    [2] Maurice G. Kendall, "The treatment of ties in ranking problems",
           Biometrika Vol. 33, No. 3, pp. 239-251. 1945.
    [3] Gottfried E. Noether, "Elements of Nonparametric Statistics",
        John Wiley & Sons, 1967.
    [4] Peter M. Fenwick, "A new data structure for cumulative frequency tables",
        Software: Practice and Experience, Vol. 24, No. 3, pp. 327-336, 1994.
    [5] Maurice G. Kendall, "Rank Correlation Methods" (4th Edition),
           Charles Griffin & Co., 1970.

    Scipy: https://github.com/scipy/scipy/blob/v1.8.1/scipy/stats/_stats_py.py#L4666-L4875

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> x2 = [1, 4, 7, 1, 0]
    >>> tau, p_value = kendall_tau(x1, x2)
    >>> tau
    -0.47140452079103173
    >>> p_value
    0.2827454599327748

    """

    tau, pvalue = stats.kendalltau(x, y, initial_lexsort=None, variant='b')

    return tau, pvalue
