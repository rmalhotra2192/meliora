import pandas as pd
from scipy.stats import t, beta, norm, binom, chisquare, chi2


def jeffreys_test(ratings, default_flag, alpha):
    """
    The Jeffreys test assesses whether the bucket PD (i.e., the PD of the
    bucket to which a given exposure is assigned) is in line with the
    empirically observed number of defaults in this bucket.

    The Jeffreys test is the most important diagnostic tool for assessing
    the calibration of the bucket probability of default (PD). However,
    it actually has a wider range of applications.

    Potentially, this test can also be applied to validate, e.g., loss given
    default (LGD) and the cure rate - the fraction of loans that recover from
    default status to non-default. Prior to examining how this test can be
    applied elsewhere, we need to discuss the challenges it presents and
    understand its mechanics, particularly: (1) its statistical definition,
    as a one-sided hypothesis test; and (2) the dynamics of the assumed
    beta distribution of the observed default rate.

    pass customer data into function, type dataframe
    columns needed: ratings, default_flag, predicted pd

    Parameters
    ----------
    default_flag : pandas series
        Boolean flag indicating whether the borrower has actually defaulted
    ratings : pandas series
        PD ratings of a counterparty

    Returns
    -------
    score : float
        Return Bayesian Error Rate

    Notes
    -----------
    Instructions for reporting the validation results of internal models, ECB, 2019
    
    """

    df3 = pd.DataFrame({'Rating': [],
                        'PD': [],
                        'N': [],
                        'D': [],
                        'Default Rate': [],
                        'P-Value': [],
                        'Pass/Fail': []})

    for rating in set(df.ratings):
        # select rating group
        df2 = df[df['ratings'] == rating]
        # the mean is calculated here and used as the pd for the rating bucket
        m = df2.prob_default.mean()
        # calculate parameters for beta distribution Beta(a,b)
        d = df2['default_flag'].sum()
        n = len(df2)
        a = d + 0.5
        b = n - d + 0.5
        # use the culmulative distribution function to calculate the std-multiplier that corresponds to the 5% interval
        p = beta.ppf(0.05, a, b)
        # results: if the rating pd is above the calculated p value, then the rating bucket passes the test
        if p <= m:
            result = 'Pass'
        else:
            result = 'Fail'

        # append the data to the created dataframe
        temp = pd.DataFrame({'Rating': [rating], 'PD': [m], 'N': [n], 'D': [
                            d], 'Default Rate': d/n, 'P-Value': [p], 'Pass/Fail': [result]})

        df3 = df3.append(temp)

    # overall
    d = df['default_flag'].sum()
    n = len(df)
    a = d + 0.5
    b = n - d + 0.5
    p = beta.ppf(0.05, a, b)
    m = df.prob_default.mean()
    # results
    if p <= m:
        result = 'Pass'
    else:
        result = 'Fail'

    overall = pd.DataFrame({'Rating': ['Overall'], 'PD': [m], 'N': [n], 'D': [
                           d], 'Default Rate': d/n, 'P-Value': [p], 'Pass/Fail': [result]})

    df3 = df3.append(overall).set_index('Rating')

    return df3
