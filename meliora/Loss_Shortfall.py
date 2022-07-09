import numpy as np
import pandas as pd
from scipy.stats import norm


def loss_shortfall(observed_lgd, elbe):

    """ Difference between observed and estimated LGDs divided by observed LGDs weighted by EAD

    https://essay.utwente.nl/61905/1/master_B._Maarse.pdf

    The current rejection areas are set as a percentage. These do not take into account that
    the LGD has a variance. If the variance of the observed loss is high the loss shortfall is
    expected to deviate more than with a low variance, because the distribution of the loss at
    default is broader. Bootstrapping with replacement confirmed this reasoning. Two
    portfolios were used with the same loss shortfall, for low variance the 95 percent
    bootstrapped confidence interval was [-0.05, -0.01] and for the high variance [-0.30,
    0.14]6
    . 
    
    This shows that the variance influence the LS and should be used when setting
    the rejection area.
    The variance of the LS is dependent on the variance of the observed losses. There is no
    linear relation, therefore distribution of predicted LS is unknown. To take the variance
    into account a distribution will be bootstrapped around the observed LS. This is done by
    sampling with replacement N observations and calculate an observed LS. This is
    repeated 1000 times to generate a distribution around the observed LS. The distribution
    will be used to test whether the expected LS of zero is within a 95/99 percent confidence
    interval of the observed LS.
    The main advantage of bootstrapping is that no assumption has to be made on the
    underlying distribution, therefore it is applicable in many cases. The main drawback is
    the computation effort needed. Another drawback is that it tends to be optimistic about
    the standard error, which results in a somewhat smaller confidence interval (Wehrens et
    al., 2000). For backtesting this means that the confidence interval might be somewhat
    conservative. 



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


    Examples
    --------
    >>> from sklearn.metrics import accuracy_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> accuracy_score(y_true, y_pred)
    0.5
    >>> accuracy_score(y_true, y_pred, normalize=False)
    2
    In the multilabel case with binary label indicators:
    >>> import numpy as np
    >>> accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
    0.5
    """

    return 1