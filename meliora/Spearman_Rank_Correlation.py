from scipy import stats


def spearman(array_1, array_2, alternative='two-sided'):
    """
    Calculate a Spearman correlation coefficient with associated p-value.

    This is a wrapper around scipy.stats.spearmanr function.

    The Spearman rank-order correlation coefficient is a nonparametric
    measure of the monotonicity of the relationship between two datasets.
    Unlike the Pearson correlation, the Spearman correlation does not
    assume that both datasets are normally distributed. Like other
    correlation coefficients, this one varies between -1 and +1 with 0
    implying no correlation. Correlations of -1 or +1 imply an exact
    monotonic relationship. Positive correlations imply that as x
    increases, so does y. Negative correlations imply that as x increases,
    y decreases.

    The p-value roughly indicates the probability of an uncorrelated 
    system producing datasets that have a Spearman correlation at least 
    as extreme as the one computed from these datasets. The p-values 
    are not entirely reliable but are probably reasonable for datasets 
    larger than 500 or so.


    Parameters
    ----------
    array_1 : pandas series
        Series containing multiple observations 
    array_2 : pandas series
        Series containing multiple observations 
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:
        * 'two-sided': the correlation is nonzero
        * 'less': the correlation is negative (less than zero)
        * 'greater':  the correlation is positive (greater than zero)

    Returns
    -------
    correlation : float or ndarray (2-D square)
        Spearman correlation matrix or correlation coefficient (if only 2
        variables are given as parameters. Correlation matrix is square with
        length equal to total number of variables (columns or rows) in ``a``
        and ``b`` combined.
    pvalue : float
        The p-value for a hypothesis test whose null hypotheisis
        is that two sets of data are uncorrelated. See `alternative` above
        for alternative hypotheses. `pvalue` has the same
        shape as `correlation`.
 


    References
    -------------
        [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
        Probability and Statistics Tables and Formulae. Chapman & Hall: New
        York. 2000.
        Section  14.7

    Examples
    --------
    >>> spearmanr([1,2,3,4,5], [5,6,7,8,7])
    SpearmanrResult(correlation=0.82078..., pvalue=0.08858...)

    """

    return stats.spearmanr(array_1, array_2, alternative='two-sided')