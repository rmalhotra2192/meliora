import pandas as pd
from sklearn.metrics import auc


def loss_capture_ratio(ead, predicted_ratings, realised_outcomes):
    """
    The loss_capture_ratio measures how well a model is able to
    rank LGDs when compared to the observed losses.

    For this approach three plots are relevant: the model loss
    capture curve, ideal loss capture curve and the random loss
    capture curve. These curves are constructed in the same way
    as the curves for the CAP. The main difference is the data,
    which is for LGDs and the LR a (continuous) percentage of the EAD,
    while for the CAP it is binary.

    The LC can be percentage weighted, which simply uses the LGD and
    LR percentages as input, while it can also be EAD weighted, which
    uses the LGD and LR multiplied with the respective EAD as input.
    The results between the two approaches can differ  if the portfolio
    is not-well balanced.

    Parameters
    ----------
    ead: pandas Series
        Exposure at Default
    predicted_ratings: pandas Series
        predicted LGD, can be ordinal or continuous
    realised_outcomes: pandas Series
        realised LGD, can be ordinal or continuous

    Returns
    -------
    LCR: scalar
        Loss Capture Ratio

    References
    ----------------

    Li, D., Bhariok, R., Keenan, S., & Santilli, S. (2009). Validation techniques 
    and performance metrics for loss given default models. 
    The Journal of Risk Model Validation, 33, 3-26.


    Examples
    --------
        >>> res = loss_capture_ratio(ead, predicted_ratings, realised_outcomes)
        >>> print(res)
    """

    # Create a dataframe
    frame = {'ead': ead,
             'predicted_ratings': predicted_ratings,
             'realised_outcomes': realised_outcomes
             }
    df = pd.DataFrame(frame)

    # Prepare data
    df['loss'] = df['ead'] * df['realised_outcomes']

    # Model loss capture curve
    df2 = df.sort_values(by='predicted_ratings', ascending=False)
    df2['cumulative_loss'] = df2.cumsum()['loss']
    df2['cumulative_loss_capture_percentage'] = df2.cumsum()['loss']/df2.loss.sum()
    auc_curve1 = auc([i for i in range(len(df2))], df2.cumulative_loss_capture_percentage)
    random_auc1 = 0.5 * len(df2) * 1

    # Ideal loss capture curve
    df3 = df.sort_values(by='realised_outcomes', ascending=False)
    df3['cumulative_loss'] = df3.cumsum()['loss']
    df3['cumulative_loss_capture_percentage'] = df3.cumsum()['loss']/df3.loss.sum()
    auc_curve2 = auc([i for i in range(len(df3))], df3.cumulative_loss_capture_percentage)
    random_auc2 = 0.5 * len(df3) * 1

    loss_capture_ratio = (auc_curve1 - random_auc1)/(auc_curve2 - random_auc2)

    return loss_capture_ratio
