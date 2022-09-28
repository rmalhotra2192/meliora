import numpy as np
from meliora.Accuracy_Ratio import accuracy_ratio


def test_accuracy_ratio_only_y():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    acc_ratio = accuracy_ratio(y_true, y_pred)

    assert acc_ratio == 1.0, "Incorrect Accuracy Ratio, It should have been 1.0"


def test_accuracy_ratio_with_normalized_input():
    y2_true = np.array([1, 0, 1])
    y2_pred = np.array([1, 0, 0])

    acc_ratio = accuracy_ratio(y2_true, y2_pred, normalize=True)
    assert (
        acc_ratio == 0.6666666666666666
    ), "Incorrect Accuracy Ratio, It should have been 0.6666666666666666"


def test_accuracy_ration_with_weights():
    y2_true = np.array([1, 0, 1])
    y2_pred = np.array([1, 0, 0])

    acc_ratio = accuracy_ratio(y2_true, y2_pred, sample_weight=[0.1, 0.2, 0.4])
    assert (
        acc_ratio == 0.4285714285714286
    ), "Incorrect Accuracy Ratio, It should have been 0.4285714285714286"
