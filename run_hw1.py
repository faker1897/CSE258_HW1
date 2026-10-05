"""Run the official HW1 interfaces on local data and check their behavior."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import inspect
import io
import json
from pathlib import Path
import platform
import random
import re
import sys
import tokenize
from unittest import mock

from dateutil import parser as date_parser
import numpy as np
from sklearn.linear_model import LogisticRegression

import homework1 as hw


BASE_DIR = Path(__file__).resolve().parent
PRECISION_KS = (1, 100, 1000, 10000)
RATING_KEYS = (
    "review/appearance",
    "review/aroma",
    "review/palate",
    "review/taste",
)


def load_books(path):
    """Read gzip-compressed JSON lines and add the runner's date field."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        row["parsed_date"] = date_parser.parse(row["date_added"])
    return rows


def load_beer(path):
    """Read Python dictionary literals without executing their contents."""
    with path.open(encoding="utf-8") as handle:
        rows = [ast.literal_eval(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Every beer review must be a dictionary.")
    return rows


def _book(length, rating, date="2024-01-01T12:00:00-08:00"):
    return {
        "review_text": "x" * length,
        "rating": rating,
        "date_added": date,
        "parsed_date": date_parser.parse(date),
    }


def _beer(length, rating):
    row = {"review/text": "x" * length, "review/overall": rating}
    row.update({key: 3.0 for key in RATING_KEYS})
    return row


def _assert_english_comments_and_docstrings(path):
    source = path.read_text(encoding="utf-8")
    cjk = re.compile(r"[\u3400-\u9fff]")
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            assert not cjk.search(token.string), (path.name, token.start)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            assert docstring is None or not cjk.search(docstring), path.name


def _assert_stub_signatures():
    notebook = json.loads((BASE_DIR / "hw1_stub.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    headers = re.findall(r"^def\s+\w+\([^\n]*\):", source, flags=re.MULTILINE)
    assert len(headers) == 12, "The official stub should expose 12 functions."
    for header in headers:
        expected = ast.parse(header + "\n    pass\n").body[0]
        function = getattr(hw, expected.name)
        signature = inspect.signature(function)
        assert list(signature.parameters) == [arg.arg for arg in expected.args.args]
        assert all(
            parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and parameter.default is inspect.Parameter.empty
            for parameter in signature.parameters.values()
        ), expected.name


def _assert_import_has_no_work():
    source = (BASE_DIR / "homework1.py").read_text(encoding="utf-8")
    forbidden = AssertionError("Importing homework1 must not perform I/O or train models.")
    with (
        mock.patch("builtins.open", side_effect=forbidden),
        mock.patch("pathlib.Path.open", side_effect=forbidden),
        mock.patch("gzip.open", side_effect=forbidden),
        mock.patch.object(LogisticRegression, "fit", side_effect=forbidden),
        mock.patch("numpy.linalg.lstsq", side_effect=forbidden),
        mock.patch("builtins.print", side_effect=forbidden),
    ):
        exec(compile(source, "homework1.py", "exec"), {"__name__": "hw1_import_check"})


def run_small_checks():
    """Check independent, small cases before evaluating the complete data."""
    checked = []
    _assert_stub_signatures()
    checked.append("official_stub_signatures")
    _assert_import_has_no_work()
    checked.append("import_without_io_training_or_printing")
    for path in (BASE_DIR / "homework1.py", Path(__file__).resolve()):
        _assert_english_comments_and_docstrings(path)
    checked.append("english_comments_and_docstrings")

    monday = _book(2, 3)
    tuesday = _book(2, 3, "2024-02-06T12:00:00-08:00")
    assert isinstance(hw.featureQ1(monday, 4), list)
    assert hw.featureQ1(monday, 4) == [1, 0.5]
    baseline = hw.featureQ2(monday, 4)
    assert isinstance(baseline, list) and len(baseline) == 19
    np.testing.assert_array_equal(baseline, [1, 0.5] + [0] * 17)
    expected = [1, 0.5] + [1, 0, 0, 0, 0, 0] + [1] + [0] * 10
    np.testing.assert_array_equal(hw.featureQ2(tuesday, 4), expected)
    assert hw.featureQ3(tuesday, 999) == [1, 2, 1, 2]
    assert hw.featureQ5({"review/text": "abc"}) == [3]
    checked.append("intercept_normalization_encoding_order_and_raw_q3_length")

    exact = [_book(0, 1), _book(1, 3), _book(2, 5)]
    theta, mse = hw.Q1(exact)
    np.testing.assert_allclose(theta, [1, 4], atol=1e-12)
    assert abs(mse) < 1e-20
    nonlinear = [_book(0, 1), _book(1, 4), _book(2, 2)]
    x2, predictions, mse2 = hw.Q2(nonlinear)
    labels = np.asarray([row["rating"] for row in nonlinear], dtype=float)
    assert not np.allclose(predictions, labels)
    expected_predictions = x2 @ np.linalg.lstsq(x2, labels, rcond=None)[0]
    np.testing.assert_allclose(predictions, expected_predictions)
    assert np.isclose(mse2, np.mean((predictions - labels) ** 2))
    x3, predictions3, mse3 = hw.Q3(nonlinear)
    assert not np.allclose(predictions3, labels)
    expected_predictions3 = x3 @ np.linalg.lstsq(x3, labels, rcond=None)[0]
    np.testing.assert_allclose(predictions3, expected_predictions3)
    assert np.isclose(mse3, np.mean((predictions3 - labels) ** 2))
    checked.append("known_least_squares_solution_and_prediction_returns")

    empty_reviews = [_book(0, 1), _book(0, 3)]
    assert hw.getMaxLen(empty_reviews) == 0
    assert hw.featureQ1(empty_reviews[0], 0) == [1, 0]
    assert hw.featureQ2(empty_reviews[0], 0) == [1, 0] + [0] * 17
    empty_theta, empty_mse = hw.Q1(empty_reviews)
    np.testing.assert_allclose(empty_theta, [2, 0], atol=1e-12)
    assert np.isclose(empty_mse, 1.0)
    checked.append("all_empty_review_normalization")

    original = _beer(3, 1)
    changed_target = dict(original, **{"review/overall": 5})
    for feature, dimension in (
        (hw._featureQ7_base, 5),
        (hw._featureQ7_extended, 8),
        (hw.featureQ7, len(hw.featureQ7(original))),
    ):
        values = feature(original)
        assert isinstance(values, list) and len(values) == dimension
        assert all(isinstance(value, (int, float, np.integer, np.floating)) for value in values)
        assert dimension in (5, 8)
        np.testing.assert_array_equal(values, feature(changed_target))
    checked.append("q7_numeric_dimensions_and_target_change_invariance")

    fixture = [
        _book(0, 1), _book(1, 3), _book(3, 4),
        _book(30, 2), _book(10, 5), _book(8, 1),
    ]
    snapshot = [dict(row) for row in fixture]
    split = len(fixture) // 2
    train, test = fixture[:split], fixture[split:]
    train_labels = np.asarray([row["rating"] for row in train], dtype=float)
    test_labels = np.asarray([row["rating"] for row in test], dtype=float)
    training_max = max(len(row["review_text"]) for row in train)
    expected_mses = []
    for feature in (hw.featureQ2, hw.featureQ3):
        train_x = np.asarray([feature(row, training_max) for row in train], dtype=float)
        test_x = np.asarray([feature(row, training_max) for row in test], dtype=float)
        model = np.linalg.lstsq(train_x, train_labels, rcond=None)[0]
        expected_mses.append(float(np.mean((test_x @ model - test_labels) ** 2)))
    original_lstsq = np.linalg.lstsq
    with mock.patch("numpy.linalg.lstsq", wraps=original_lstsq) as fit_spy:
        actual_mses = hw.Q4(fixture)
    assert len(fit_spy.call_args_list) == 2
    for call in fit_spy.call_args_list:
        np.testing.assert_array_equal(call.args[1], train_labels)
        assert len(call.args[0]) == split
    np.testing.assert_allclose(actual_mses, expected_mses)
    assert fixture == snapshot, "Q4 must preserve the supplied order and data."
    altered = [dict(row) for row in fixture]
    for row in altered[split:]:
        row["rating"] += 10
    with mock.patch("numpy.linalg.lstsq", wraps=original_lstsq) as altered_spy:
        hw.Q4(altered)
    for before, after in zip(fit_spy.call_args_list, altered_spy.call_args_list):
        np.testing.assert_array_equal(before.args[0], after.args[0])
        np.testing.assert_array_equal(before.args[1], after.args[1])
    checked.append("q4_training_only_scaling_labels_order_and_test_label_isolation")

    actual = np.array([True, True, False, False])
    predicted = np.array([True, False, True, False])
    counts = hw._classification_metrics(actual, predicted)
    assert counts == (1, 1, 1, 1, 0.5)
    assert all(type(value) is int for value in counts[:4])
    assert type(counts[4]) is float
    tied = hw._precision_at_k(
        np.array([False, True, True, False]), np.array([0.8, 0.8, 0.7, 0.1]), (1, 2, 4)
    )
    np.testing.assert_allclose(tied, [0, 0.5, 0.5])
    boundary = [_beer(1, 3.9), _beer(2, 4), _beer(3, 4.5), _beer(4, 1)]
    classifier, matrix, labels = hw._fit_classifier(boundary, hw.featureQ5)
    np.testing.assert_array_equal(labels, [False, True, True, False])
    assert matrix.shape == (4, 1)
    assert classifier.class_weight == "balanced" and classifier.fit_intercept
    checked.append("label_threshold_balanced_classifier_confusion_ber_and_stable_precision")
    return checked


def _finite(value):
    return bool(np.all(np.isfinite(np.asarray(value, dtype=float))))


def run_full_checks(books, beer, outputs, arrays):
    """Validate full-data outputs already computed by the runner."""
    assert len(books) == 10000 and len(beer) == 50000
    theta, x2, predictions2, x3, predictions3 = arrays
    assert isinstance(theta, np.ndarray) and theta.shape == (2,)
    assert isinstance(x2, np.ndarray) and x2.shape == (len(books), 19)
    assert isinstance(x3, np.ndarray) and x3.shape == (len(books), 4)
    assert isinstance(predictions2, np.ndarray) and predictions2.shape == (len(books),)
    assert isinstance(predictions3, np.ndarray) and predictions3.shape == (len(books),)
    assert all(_finite(array) for array in arrays)
    for question in ("Q1", "Q2", "Q3"):
        assert type(outputs[question]["mse"]) is float
        assert _finite(outputs[question]["mse"]) and outputs[question]["mse"] >= 0
    assert _finite(list(outputs["Q4"].values()))
    for question in ("Q5", "Q7"):
        metrics = outputs[question]["metrics"]
        assert all(type(metrics[key]) is int for key in ("TP", "TN", "FP", "FN"))
        assert sum(metrics[key] for key in ("TP", "TN", "FP", "FN")) == len(beer)
        assert 0 <= metrics["BER"] <= 1
        assert np.isclose(
            metrics["BER"],
            0.5 * (
                metrics["FN"] / (metrics["TP"] + metrics["FN"])
                + metrics["FP"] / (metrics["TN"] + metrics["FP"])
            ),
        )
    precisions = outputs["Q6"]["precision"]
    assert outputs["Q6"]["k"] == list(PRECISION_KS)
    assert len(precisions) == 4 and all(type(value) is float for value in precisions)
    assert _finite(precisions) and all(0 <= value <= 1 for value in precisions)
    selected = outputs["Q7"]["selected_candidate"]
    feature = hw._featureQ7_base if selected == "base" else hw._featureQ7_extended
    for row in beer:
        actual = hw.featureQ7(row)
        assert isinstance(actual, list)
        np.testing.assert_allclose(actual, feature(row), rtol=0, atol=0)
    assert np.isclose(
        outputs["Q7"]["metrics"]["BER"], outputs["Q7"]["candidate_ber"][selected]
    )
    return [
        "complete_dataset_counts_and_shapes",
        "finite_outputs_and_scalar_return_types",
        "full_confusion_counts_ber_and_precision_ranges",
        "selected_q7_feature_matches_best_fixed_candidate",
    ]


def _metrics(values):
    return dict(zip(("TP", "TN", "FP", "FN", "BER"), values))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment():
    versions = {}
    for package in ("numpy", "scikit-learn", "scipy", "python-dateutil"):
        versions[package] = importlib.metadata.version(package)
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def main(argv=None):
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--check", action="store_true", help="Run small and full-data checks.")
    argument_parser.add_argument(
        "--output", type=Path, default=BASE_DIR / "results.json", help="Save actual results as JSON."
    )
    arguments = argument_parser.parse_args(argv)
    checked = run_small_checks() if arguments.check else []
    books = load_books(BASE_DIR / "datasets" / "fantasy_10000.json.gz")
    beer = load_beer(BASE_DIR / "datasets" / "beer_50000.json")
    shuffled = books[:]
    random.Random(0).shuffle(shuffled)

    theta1, mse1 = hw.Q1(books)
    x2, predictions2, mse2 = hw.Q2(books)
    x3, predictions3, mse3 = hw.Q3(books)
    mse4_onehot, mse4_numeric = hw.Q4(shuffled)
    metrics5 = hw.Q5(beer, hw.featureQ5)
    precisions = hw.Q6(beer)
    candidate_models = {}
    candidate_ber = {}
    for name, feature in (
        ("base", hw._featureQ7_base),
        ("extended", hw._featureQ7_extended),
    ):
        model, matrix, labels = hw._fit_classifier(beer, feature)
        candidate_models[name] = model
        candidate_ber[name] = hw._classification_metrics(labels, model.predict(matrix))[-1]
    selected = "extended" if candidate_ber["extended"] < candidate_ber["base"] else "base"
    metrics7 = hw.Q5(beer, hw.featureQ7)
    absolute_improvement = float(metrics5[-1] - metrics7[-1])
    relative_improvement = float(absolute_improvement / metrics5[-1]) if metrics5[-1] else 0.0
    target_met = bool(absolute_improvement >= 0.03 and relative_improvement >= 0.03)

    outputs = {
        "Q1": {"theta": theta1.tolist(), "mse": mse1},
        "Q2": {
            "shape": list(x2.shape), "first_features": x2[0].tolist(),
            "first_prediction": float(predictions2[0]), "mse": mse2,
        },
        "Q3": {
            "shape": list(x3.shape), "first_features": x3[0].tolist(),
            "first_prediction": float(predictions3[0]), "mse": mse3,
        },
        "Q4": {"onehot_test_mse": mse4_onehot, "numeric_test_mse": mse4_numeric},
        "Q5": {"metrics": _metrics(metrics5)},
        "Q6": {"k": list(PRECISION_KS), "precision": list(precisions)},
        "Q7": {
            "metrics": _metrics(metrics7), "candidate_ber": candidate_ber,
            "candidate_training": {
                name: {"n_iter": model.n_iter_.tolist(), "max_iter": model.max_iter}
                for name, model in candidate_models.items()
            },
            "selected_candidate": selected, "first_features": hw.featureQ7(beer[0]),
            "feature_dimension": len(hw.featureQ7(beer[0])),
            "coefficients": candidate_models[selected].coef_[0].tolist(),
            "intercept": float(candidate_models[selected].intercept_[0]),
            "absolute_ber_improvement": absolute_improvement,
            "relative_ber_improvement": relative_improvement,
            "local_target_met": target_met,
        },
    }
    if arguments.check:
        checked.extend(run_full_checks(
            books, beer, outputs, (theta1, x2, predictions2, x3, predictions3)
        ))
    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(),
        "original_notebook_sha256": {
            name: _sha256(BASE_DIR / name)
            for name in ("hw1_stub.ipynb", "hw1_runner.ipynb")
        },
        "data": {
            "books": len(books), "beer": len(beer),
            "zero_book_ratings": sum(row["rating"] == 0 for row in books),
            "maximum_book_review_length": hw.getMaxLen(books),
            "positive_beer_labels": sum(row["review/overall"] >= 4 for row in beer),
            "negative_beer_labels": sum(row["review/overall"] < 4 for row in beer),
            "q4_shuffle_seed": 0, "q4_train_count": len(books) // 2,
        },
        "classifier_parameters": {
            "class_weight": "balanced", "fit_intercept": True, "solver": "lbfgs",
            "C": 1.0, "tol": 1e-4, "max_iter": 1000,
            "convergence_retry_max_iter": 10000,
        },
        "outputs": outputs,
        "checks": {
            "requested": arguments.check, "passed": checked,
            "models_converged": True, "q7_local_target_met": target_met,
            "status": "passed" if target_met else "failed_q7_target",
        },
    }
    arguments.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Loaded {len(books)} book reviews and {len(beer)} beer reviews.")
    for name in ("Q1", "Q2", "Q3"):
        print(f"{name}: MSE = {outputs[name]['mse']:.10f}")
    print(f"Q1: theta = {theta1.tolist()}")
    print(f"Q4: one-hot MSE = {mse4_onehot:.10f}; numeric MSE = {mse4_numeric:.10f}")
    print(f"Q5: {_metrics(metrics5)}")
    print(f"Q6: Precision@{list(PRECISION_KS)} = {list(precisions)}")
    print(f"Q7: candidate BER = {candidate_ber}; selected = {selected}")
    print(
        f"Q7: BER = {metrics7[-1]:.10f}; absolute improvement = {absolute_improvement:.6f}; "
        f"relative improvement = {relative_improvement:.2%}"
    )
    if arguments.check:
        print(f"Checks: {len(checked)} groups passed.")
    print(f"Saved results: {arguments.output.resolve()}")
    if not target_met:
        print("FAILED: Q7 did not meet both local BER improvement targets.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
