# various tests for utils functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


from utils import create_labels, create_random_predictions, \
    create_perfect_predictions, calculate_roc_auc, calculate_pr_auc, \
    plot_roc_curve, plot_pr_curve, create_noisy_predictions


def test_create_dataset_array():
    data = create_labels()
    assert type(data) == np.ndarray


def test_create_dataset_nsize():
    data = create_labels(n=10)
    assert len(data) == 10


def test_create_dataset_badrate():
    data = create_labels(n=10000, bad_rate=0.5)
    assert sum(data) > 10000*.4 and sum(data) < 10000*.60


def test_create_random_predictions_nsize():
    labels = create_labels()
    predictions = create_random_predictions(labels)
    assert len(labels) == len(predictions)


def test_create_random_predictions_isfloat():
    labels = create_labels(n=10)
    predictions = create_random_predictions(labels)
    for p in predictions:
        assert type(p) == np.float64


def test_create_perfect_predictions_nsize():
    labels = create_labels()
    predictions = create_perfect_predictions(labels)
    assert len(labels) == len(predictions)


def test_create_perfect_predictions_isfloat():
    labels = create_labels(n=10)
    predictions = create_perfect_predictions(labels)
    for p in predictions:
        assert type(p) == np.float64


def test_noisy_predictions_nsize():
    labels = create_labels(n=1000)
    predictions = create_noisy_predictions(labels)
    assert len(labels) == len(predictions)


def test_noisy_predictions_isfloat():
    labels = create_labels(n=1000)
    predictions = create_noisy_predictions(labels)
    for p in predictions:
        assert type(p) == np.float64


def test_noisy_predictions_between_0_and_1():
    labels = create_labels(n=1000)
    predictions = create_noisy_predictions(labels, noise_level=0.4)
    for p in predictions:
        assert p >= 0.0 and p <= 1.0


def test_roc_auc_random():
    labels = create_labels(n=1000)
    predictions = create_random_predictions(labels)
    auc = calculate_roc_auc(labels, predictions)
    assert auc > 0.45 and auc < 0.55


def test_roc_auc_perfect():
    labels = create_labels(n=1000)
    predictions = create_perfect_predictions(labels)
    auc = calculate_roc_auc(labels, predictions)
    assert auc == 1.0


def test_pr_auc_random():
    bad_rate = 0.2
    labels = create_labels(n=10000, bad_rate=bad_rate)
    predictions = create_random_predictions(labels)
    auc = calculate_pr_auc(labels, predictions)
    assert auc > bad_rate - bad_rate*0.33 and auc < bad_rate + bad_rate*0.33


def test_pr_auc_perfect():
    labels = create_labels(n=1000)
    predictions = create_perfect_predictions(labels)
    auc = calculate_pr_auc(labels, predictions)
    assert auc == 1.0


def test_auc_roc_plot():
    labels = create_labels(n=1000)
    predictions = create_perfect_predictions(labels)
    plot = plot_roc_curve(labels, predictions)
    assert type(plot) == plt.Figure


def test_auc_pr_plot():
    labels = create_labels(n=1000)
    predictions = create_perfect_predictions(labels)
    plot = plot_pr_curve(labels, predictions)
    assert type(plot) == plt.Figure
