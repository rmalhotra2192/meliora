from scipy import stats

def somersd(array_1, array_2, alternative='two-sided'):
    """
    Calculates Somers' D, an asymmetric measure of ordinal association.

    This is a wrapper around scipy.stats.somersd function.

    Somers' :math:`D` is a measure of the correspondence between two rankings. 
    It considers the difference between the number of concordant 
    and discordant pairs in two rankings and is  normalized such that values
    close  to 1 indicate strong agreement and values close to -1 indicate
    strong disagreement. 

    Parameters
    ----------
    x: array_like
        1D array of rankings, treated as the (row) independent variable.
        Alternatively, a 2D contingency table.
    y: array_like, optional
        If `x` is a 1D array of rankings, `y` is a 1D array of rankings of the
        same length, treated as the (column) dependent variable.
        If `x` is 2D, `y` is ignored.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:
        * 'two-sided': the rank correlation is nonzero
        * 'less': the rank correlation is negative (less than zero)
        * 'greater':  the rank correlation is positive (greater than zero)

    Returns
    -------
    res : SomersDResult
        A `SomersDResult` object with the following fields:
            correlation : float
               The Somers' :math:`D` statistic.
            pvalue : float
               The p-value for a hypothesis test whose null
               hypothesis is an absence of association, :math:`D=0`.
               See notes for more information.
            table : 2D array
               The contingency table formed from rankings `x` and `y` (or the
               provided contingency table, if `x` is a 2D array)

    References
    ----------
    [1] Robert H. Somers, "A New Asymmetric Measure of Association for
           Ordinal Variables", *American Sociological Review*, Vol. 27, No. 6,
           pp. 799--811, 1962.
    [2] Morton B. Brown and Jacqueline K. Benedetti, "Sampling Behavior of
           Tests for Correlation in Two-Way Contingency Tables", *Journal of
           the American Statistical Association* Vol. 72, No. 358, pp.
           309--315, 1977.
    [3] SAS Institute, Inc., "The FREQ Procedure (Book Excerpt)",
           *SAS/STAT 9.2 User's Guide, Second Edition*, SAS Publishing, 2009.
    [4] Laerd Statistics, "Somers' d using SPSS Statistics", *SPSS
           Statistics Tutorials and Statistical Guides*,
           https://statistics.laerd.com/spss-tutorials/somers-d-using-spss-statistics.php,
           Accessed July 31, 2020.

    Examples
    --------
    >>> table = [[27, 25, 14, 7, 0], [7, 14, 18, 35, 12], [1, 3, 2, 7, 17]]
    >>> res = somersd(table)
    >>> res.statistic
    0.6032766111513396
    >>> res.pvalue
    1.0007091191074533e-27
    
    """

    return stats.spearmanr(array_1, array_2, alternative='two-sided')

