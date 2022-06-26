import numpy as np
import pandas as pd


def rating_concentration(initial_ratings, final_ratings):
    """ Assess whether dispersion of the PD rating grades is adequate

    The objective of this validation tool is to assess whether rating grades have
    meaningful dispersion - i.e. to benchmark the current concentration level against the
    concentration level measured at the time of the initial validation39 in the course of
    the model's development. The level of concentration is calculated both in terms of the
    percentage of customers and in terms of exposure.

    Parameters
    ----------
    initial_ratings_col: Pandas series
        series with initial ratings
    final_ratings_col: Pandas series
        series with final ratings

    Returns
    -------
    H_init: float
        Initial Herfindahl index
    H_curr: float
        Final Herfindahl index
    p_value: float
        p-value of the test


    Warning
    -------

    See Also
    --------

    Index.unique : Return unique values from an Index.
    Series.unique : Return unique values of Series object.

    Notes
    -----------

    The objective of this validation tool is to assess whether rating grades have
    meaningful dispersion - i.e. to benchmark the current concentration level against the
    concentration level measured at the time of the initial validation39 in the course of
    the model's development. The level of concentration is calculated both in terms of the
    percentage of customers and in terms of exposure.

    Comparison of the Herfindahl Index at the beginning of the relevant observation
    period and the Herfindahl Index at the time of the initial validation during
    development via hypothesis testing based on a normal approximation assuming a
    deterministic Herfindahl Index at the time of the model's development. The null
    hypothesis of the test40 is that the current Herfindahl Index is lower than the
    Herfindahl Index at the time of development.

    The test should be performed at number-weighted and exposure-weighted level.


    Observations are assumed to be independent.
    This function can be used for both performing and non-performing LGDs.

    References
    -----------
    
    For more information see "Instructions for reporting the validation results
    of internal models.", ECB, February 2019

    ExamplesXX
    ----------
    .. code-block:: python

        >>> res = migration_matrix_statistics(df=df, initial_ratings_col='ratings', final_ratings_col='ratings2')
        >>> print(res)
    """

    # Check the data for missing values
    if initial_ratings.isnull().values.any() or final_ratings.isnull().values.any():
        raise ValueError('Missing values detected')
    if len(initial_ratings) != len(final_ratings):
        raise ValueError('The length of the rating series does not match')

    # Create a crosstab for initial and final ratings
    N_init = pd.crosstab(initial_ratings, initial_ratings)
    N_curr = pd.crosstab(final_ratings, final_ratings)

    # Calculate the coefficient of variation
    K = len(set(initial_ratings))
    R_init = list(N_init.sum(axis=1) / N_init.sum(axis=1).sum())
    R_curr = list(N_curr.sum() / N_curr.sum().sum())

    CV_init = CV_curr = 0
    for i in range(1, K+1):
        CV_init += (R_init[i-1] - 1/K)**2
    CV_init = np.sqrt(K * CV_init)

    for i in range(1, K+1):
        CV_curr += (R_curr[i - 1] - 1 / K)**2
    CV_curr = np.sqrt(K * CV_curr)

    # Calculate the Herfindahl Index
    HI_init = 1 + np.log((CV_init**2 + 1) / K) / np.log(K)
    HI_curr = 1 + np.log((CV_curr**2 + 1) / K) / np.log(K)

    # Calculate the p-value
    p1 = np.sqrt(K - 1) * (CV_curr - CV_init)
    p2 = np.sqrt(CV_curr**2 * (0.5 + CV_curr**2))
    p_value = 1 - norm.cdf(p1 / p2)

    return HI_curr, HI_init, p_value
