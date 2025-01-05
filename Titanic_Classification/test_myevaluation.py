# pylint: skip-file
"""test_myevaluation.py

@author gsprint23
Note: do not modify this file
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from mysklearn import myevaluation

# in-class binary example for precision, recall, f1score
win_lose_y_true = ["win"] * 20 + ["lose"] * 20
win_lose_y_pred = ["win"] * 18 + ["lose"] * 2 + ["win"] * 12 + ["lose"] * 8

# bramer ch. 12 binary example for precision, recall, f1
P = 60
N = 40
pos_neg_y_true = ["+"] * P + ["-"] * N
pos_neg_perfect_classifier_y_pred = ["+"] * P + ["-"] * N
pos_neg_worst_possible_classifier_y_pred = ["-"] * P + ["+"] * N
pos_neg_ultra_liberal_classifier_y_pred = ["+"] * (P + N)
pos_neg_ultra_conservative_classifier_y_pred = ["-"] * (P + N)

# note: order is actual/received student value, expected/solution
def test_binary_precision_score():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    # bramer ch. 12 binary examples
    labels = ["+", "-"]
    precision = myevaluation.binary_precision_score(pos_neg_y_true, pos_neg_perfect_classifier_y_pred, labels=labels, pos_label="+")
    assert np.isclose(precision, 1.0)
    precision = myevaluation.binary_precision_score(pos_neg_y_true, pos_neg_worst_possible_classifier_y_pred, labels=labels, pos_label="+")
    assert np.isclose(precision, 0.0)
    precision = myevaluation.binary_precision_score(pos_neg_y_true, pos_neg_ultra_liberal_classifier_y_pred, labels=labels, pos_label="+")
    assert np.isclose(precision, (P / (P + N)))
    precision = myevaluation.binary_precision_score(pos_neg_y_true, pos_neg_ultra_conservative_classifier_y_pred, labels=labels, pos_label="+")
    assert np.isclose(precision, 0.0) # "Precision is not applicable as TP + FP = 0"

    # in-class binary examples
    labels = ["win", "lose"]
    for label in labels: # treat each label as the pos_label
        precision_solution = precision_score(win_lose_y_true, win_lose_y_pred, labels=labels, pos_label=label, average="binary")
        precision = myevaluation.binary_precision_score(win_lose_y_true, win_lose_y_pred, labels=labels, pos_label=label)
        assert np.isclose(precision, precision_solution)

def test_binary_recall_score():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    # bramer ch. 12 binary examples
    labels = ["+", "-"]
    recall = myevaluation.binary_recall_score(pos_neg_y_true, pos_neg_perfect_classifier_y_pred, labels=labels, pos_label="+")
    assert np.isclose(recall, 1.0)
    recall = myevaluation.binary_recall_score(pos_neg_y_true, pos_neg_worst_possible_classifier_y_pred, labels=labels, pos_label="+")
    assert np.isclose(recall, 0.0)
    recall = myevaluation.binary_recall_score(pos_neg_y_true, pos_neg_ultra_liberal_classifier_y_pred, labels=labels, pos_label="+")
    assert np.isclose(recall, 1.0)
    recall = myevaluation.binary_recall_score(pos_neg_y_true, pos_neg_ultra_conservative_classifier_y_pred, labels=labels, pos_label="+")
    assert np.isclose(recall, 0.0)

    # in-class binary examples
    labels = ["win", "lose"]
    for label in labels: # treat each label as the pos_label
        recall_solution = recall_score(win_lose_y_true, win_lose_y_pred, labels=labels, pos_label=label, average="binary")
        recall = myevaluation.binary_recall_score(win_lose_y_true, win_lose_y_pred, labels=labels, pos_label=label)
        assert np.isclose(recall, recall_solution)

def test_binary_f1_score():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    # bramer ch. 12 binary examples
    labels = ["+", "-"]
    f1 = myevaluation.binary_f1_score(pos_neg_y_true, pos_neg_perfect_classifier_y_pred, labels=labels, pos_label="+")
    assert np.isclose(f1, 1.0)
    f1 = myevaluation.binary_f1_score(pos_neg_y_true, pos_neg_worst_possible_classifier_y_pred, labels=labels, pos_label="+")
    assert np.isclose(f1, 0.0) # "F1 Score is not applicable as Precision + Recall = 0"
    f1 = myevaluation.binary_f1_score(pos_neg_y_true, pos_neg_ultra_liberal_classifier_y_pred, labels=labels, pos_label="+")
    assert np.isclose(f1, (2 * P / (2 * P + N)))
    f1 = myevaluation.binary_f1_score(pos_neg_y_true, pos_neg_ultra_conservative_classifier_y_pred, labels=labels, pos_label="+")
    assert np.isclose(f1, 0.0) # "F1 Score is not applicable as Precision + Recall = 0"

    # in-class binary examples
    labels = ["win", "lose"]
    for label in labels: # treat each label as the pos_label
        f1_solution = f1_score(win_lose_y_true, win_lose_y_pred, labels=labels, pos_label=label, average="binary")
        f1 = myevaluation.binary_f1_score(win_lose_y_true, win_lose_y_pred, labels=labels, pos_label=label)
        assert np.isclose(f1, f1_solution)
