import pandas as pd
import numpy as np
from sklearn.metrics import auc


def clar(predicted_ratings, realised_outcomes):
    """
    CLAR serves as a measure of ranking ability against LGD risk

    The cumulative LGD accuracy ratio (CLAR) curve can be treated as
    the equivalent of the Cumulative Accuracy Profile (CAP) curve. This
    test compares the cumulative percentage of correctly assigned realized
    LGD and the cumulative rate of observations in the predicted LGD bands.

    Parameters
    ----------
    predicted_ratings: pandas Series
        predicted LGD, can be ordinal or continuous
    realised_outcomes: pandas Series
        realised LGD, can be ordinal or continuous

    Returns
    -------
    clar: scalar
        Cumulative LGD Accuracy Ratio

    References
    --------------
    [1] Ozdemir, B., Miu, P., 2009. Basel II Implementation.
    A Guide to Developing and Validating a Compliant Internal Risk Rating
    System. McGraw-Hill, USA.

    [2] See also: https://rdrr.io/cran/VUROCS/man/clar.html

    Examples
    --------
        >>> res = clar(predicted_ratings, realised_outcomes)
        >>> print(res)

    """

    # Create a dataframe
    frame = {
             'predicted_ratings': predicted_ratings,
             'realised_outcomes': realised_outcomes
             }
    df = pd.DataFrame(frame)

    # Calculate CLAR
    x_s = [0]
    x_values = [0]
    y_values = [0]

    for i, j in enumerate(list(set(df.bucket))[::-1]):
        x = (df.predicted_ratings == j).sum()
        x_bucket = df.sort_values(by='realised_outcomes', ascending=False)[x_s[i]:x_s[i]+x]
        x_value = x/len(df)
        y_value = (x_bucket.bucket == j).sum()/len((x_bucket.bucket == j))
        x_values.append(x_value)
        y_values.append(y_value)
        x_s.append(x+1)

    new_xvalues = list(np.cumsum(x_values))
    new_yvalues = list(np.cumsum(y_values))

    model_auc = auc(new_xvalues, new_yvalues)
    clar = 2*model_auc

    return clar
