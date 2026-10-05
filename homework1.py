"""Homework 1 regression and classification functions."""

import math
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def getMaxLen(dataset):
    """Return the maximum number of characters in a book review."""
    if not dataset:
        raise ValueError("The dataset must contain at least one review.")
    return max(len(datum["review_text"]) for datum in dataset)


def _scaled_length(datum, maxLen):
    """Map an all-empty collection to zero without dividing by zero."""
    return len(datum["review_text"]) / maxLen if maxLen else 0.0


def featureQ1(datum, maxLen):
    """Return the intercept and normalized review length."""
    return [1, _scaled_length(datum, maxLen)]


def _regression_data(dataset, feature_function, maxLen):
    X = np.asarray(
        [feature_function(datum, maxLen) for datum in dataset], dtype=float
    )
    y = np.asarray([datum["rating"] for datum in dataset], dtype=float)
    return X, y


def _least_squares(X, y):
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    predictions = X @ theta
    mse = float(np.mean((predictions - y) ** 2))
    return theta, predictions, mse


def Q1(dataset):
    """Fit the length model and return its parameters and training MSE."""
    X, y = _regression_data(dataset, featureQ1, getMaxLen(dataset))
    theta, _, mse = _least_squares(X, y)
    return theta, mse


def featureQ2(datum, maxLen):
    """Use Monday and January as the omitted one-hot reference categories."""
    date = datum["parsed_date"]
    weekdays = [int(date.weekday() == day) for day in range(1, 7)]
    months = [int(date.month == month) for month in range(2, 13)]
    return [1, _scaled_length(datum, maxLen)] + weekdays + months


def Q2(dataset):
    """Return the one-hot design matrix, predicted ratings, and training MSE."""
    X, y = _regression_data(dataset, featureQ2, getMaxLen(dataset))
    _, predictions, mse = _least_squares(X, y)
    return X, predictions, mse


def featureQ3(datum, maxLen):
    """Keep the stub signature while using raw length as written in Q3."""
    date = datum["parsed_date"]
    return [1, len(datum["review_text"]), date.weekday(), date.month]


def Q3(dataset):
    """Return the numeric-date design matrix, predicted ratings, and MSE."""
    X, y = _regression_data(dataset, featureQ3, getMaxLen(dataset))
    _, predictions, mse = _least_squares(X, y)
    return X, predictions, mse


def Q4(dataset):
    """Fit on the supplied first half and evaluate on the second half."""
    if len(dataset) < 2:
        raise ValueError("Q4 requires at least two reviews.")
    split = len(dataset) // 2
    train, test = dataset[:split], dataset[split:]
    maxLen = getMaxLen(train)
    errors = []
    for feature_function in (featureQ2, featureQ3):
        X_train, y_train = _regression_data(train, feature_function, maxLen)
        X_test, y_test = _regression_data(test, feature_function, maxLen)
        theta, _, _ = _least_squares(X_train, y_train)
        errors.append(float(np.mean((X_test @ theta - y_test) ** 2)))
    return errors[0], errors[1]


def featureQ5(datum):
    """Return raw beer-review length; the classifier supplies the intercept."""
    return [len(datum["review/text"])]


def _fit_classifier(dataset, feat_func):
    if not dataset:
        raise ValueError("The dataset must contain at least one review.")
    X = np.asarray([feat_func(datum) for datum in dataset], dtype=float)
    y = np.asarray([datum["review/overall"] >= 4 for datum in dataset], dtype=bool)
    if np.unique(y).size != 2:
        raise ValueError("Classification requires both positive and negative reviews.")

    # Retry only the iteration limit, keeping the statistical model unchanged.
    for max_iter in (1000, 10000):
        model = LogisticRegression(
            class_weight="balanced",
            fit_intercept=True,
            solver="lbfgs",
            C=1.0,
            tol=1e-4,
            max_iter=max_iter,
        )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)
        converged = True
        for warning in recorded:
            if issubclass(warning.category, ConvergenceWarning):
                converged = False
            else:
                warnings.warn(str(warning.message), warning.category, stacklevel=2)
        if converged:
            return model, X, y
    raise RuntimeError("Logistic regression did not converge after 10000 iterations.")


def _classification_metrics(y, predictions):
    y = np.asarray(y, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    TP = int(np.count_nonzero(y & predictions))
    TN = int(np.count_nonzero(~y & ~predictions))
    FP = int(np.count_nonzero(~y & predictions))
    FN = int(np.count_nonzero(y & ~predictions))
    if TP + FN == 0 or TN + FP == 0:
        raise ValueError("BER requires both positive and negative labels.")
    BER = float(0.5 * (FN / (TP + FN) + FP / (TN + FP)))
    return TP, TN, FP, FN, BER


def Q5(dataset, feat_func):
    """Train balanced logistic regression and evaluate the supplied reviews."""
    model, X, y = _fit_classifier(dataset, feat_func)
    return _classification_metrics(y, model.predict(X))


def _precision_at_k(y, scores, ks):
    """Use stable descending ranks and all available rows when K exceeds N."""
    y = np.asarray(y, dtype=bool)
    if not y.size:
        raise ValueError("Precision requires at least one label.")
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return [float(np.mean(y[order[:k]])) for k in ks]


def Q6(dataset):
    """Return positive-class precision at 1, 100, 1000, and 10000."""
    model, X, y = _fit_classifier(dataset, featureQ5)
    positive_column = int(np.flatnonzero(model.classes_ == True)[0])
    scores = model.predict_proba(X)[:, positive_column]
    return _precision_at_k(y, scores, (1, 100, 1000, 10000))


def _featureQ7_base(datum):
    return [
        math.log1p(len(datum["review/text"])),
        float(datum["review/appearance"]),
        float(datum["review/aroma"]),
        float(datum["review/palate"]),
        float(datum["review/taste"]),
    ]


def _featureQ7_extended(datum):
    features = _featureQ7_base(datum)
    ratings = features[1:]
    return features + [sum(ratings) / len(ratings), min(ratings), max(ratings)]


def featureQ7(datum):
    """Return the fixed improved features without accessing the target label."""
    return _featureQ7_base(datum)
