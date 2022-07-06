import numpy as np


def elbe_t_test(observed_lgd, elbe):
    """
    The objective of this validation tool (ELBE back-testing using 1
    a one-sample t-test for paired observations) is to assess the 
    predictive ability of ELBE at portfolio level, as well as at 
    grade/pool or segment level, at various reference points in default.

    The one-sample t-test for paired observations compares ELBE with 
    realised LGD under the null hypothesis that ELBE is equal to 
    realised LGD (two-sided hypothesis test), assuming independent 
    observations. Under the null hypothesis, the test statistic is 
    asymptotically Student-t distributed with (N - 1) degrees of 
    freedom, where N denotes the number of facilities (back-testing). 

    Parameters
    ----------
    observed_lgd: pandas Series
        realised LGD, float
    elbe: pandas Series
        ELBE for each facility, float

    Returns
    -------
    t_stat: scalar
        t-statistics
    p_value: scalar
        p-value

    References
    --------------
    [1] https://www.bankingsupervision.europa.eu/banking/tasks/internal_models/shared/pdf/instructions_validation_reporting_credit_risk.en.pdf


    Examples
    --------
        >>> res = elbe_t_test(observed_lgd, elbe)
        >>> print(res)

    """

    N = len(observed_lgd)
    error = observed_lgd - elbe
    mean_error = error.mean()
    num = np.sqrt(N)*mean_error
    s2 = (((observed_lgd - elbe) - mean_error)**2).sum()/(N-1)
    t_stat = num/np.sqrt(s2)
    p_value = 2*(1 - t.cdf(abs(t_stat), df=N-1))

    return t_stat, p_value 
