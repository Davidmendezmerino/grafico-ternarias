"""Tests for the ternary plot example script."""

import math

import pytest

# Skip the entire test module if the plotting dependencies are not available.
pytest.importorskip("matplotlib")
pytest.importorskip("ternary")

import ternary_plot_example as example


def test_random_simplex_point_on_simplex():
    point = example.random_simplex_point()
    assert len(point) == 3
    assert all(component >= 0 for component in point)
    assert math.isclose(sum(point), 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_generate_dataset_size_and_simplex_membership():
    compositions, predictions = example.generate_dataset(n_samples=5)
    assert len(compositions) == len(predictions) == 5
    for composition in compositions:
        assert len(composition) == 3
        assert all(component >= 0 for component in composition)
        assert math.isclose(sum(composition), 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_normalize_predictions_range_and_bounds():
    data = [2.0, 4.0, 6.0]
    normalized = example.normalize_predictions(data)
    assert normalized[0] == pytest.approx(0.0)
    assert normalized[-1] == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in normalized)


def test_normalize_predictions_identical_values_returns_zeros():
    data = [3.5, 3.5, 3.5]
    normalized = example.normalize_predictions(data)
    assert normalized == [0.0, 0.0, 0.0]


def test_plot_ternary_scatter_creates_output_file(tmp_path):
    compositions = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    predictions = [0.1, 0.2, 0.3]
    output_path = tmp_path / "test_plot.png"

    example.plot_ternary_scatter(compositions, predictions, output_path)

    assert output_path.exists()
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
