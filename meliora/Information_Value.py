import pandas as pd


def calc_iv(df, feature, target, pr=0):
    """
    A numerical value that quantifies the predictive power of an independent 
    variable in capturing the binary dependent variable.

    Weight of evidence (WOE) is a measure of how much the evidence supports or 
    undermines a hypothesis. WOE measures the relative risk of an attribute of 
    binning level. The value depends on whether the value of the target variable 
    is a nonevent or an event.

    The information value (IV) is a weighted sum of the WOE of the 
    characteristic's attributes. The weight is the difference between the 
    conditional probability of an attribute for an event and the conditional 
    probability of that attribute for a nonevent. 

    An information value can be any real number. Generally speaking, the higher 
    the information value, the more predictive an attribute is likely to be.


    Parameters
    ----------
    df : Pandas dataframe
        Contains information on the the feature and target variable
    feature : string
        independent variable
    feature : string
        dependent variable

    Returns
    -------
    iv : float
       Information Value.

    References
    --------------
    -  https://www.lexjansen.com/mwsug/2013/AA/MWSUG-2013-AA14.pdf.
    -  https://documentation.sas.com/doc/en/vdmmlcdc/8.1/casstat/viyastat_binning_details02.htm.


    Examples
    --------
    >>> iv = calc_iv(df, feature, target, pr=0)
    >>> iv
    -0.47140452079103173
    
    """

    lst = []

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Bad'])
    data = data[data['Bad'] > 0]

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    data['IV'] = (data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])).sum()

    data = data.sort_values(by=['Variable', 'Value'], ascending=True)

    return data['IV'].values[0]