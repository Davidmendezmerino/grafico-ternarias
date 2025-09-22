"""Example ternary scatter plot for ternary mixture predictions.

This script generates synthetic predictions for 120 ternary mixtures and
visualizes them on a ternary scatter plot using the `python-ternary`
package. Each point represents a mixture whose three coordinates sum to 1.
The color of the point encodes a model response (e.g., predicted yield).

Prerequisites
-------------
Install dependencies with:

    pip install matplotlib python-ternary numpy

Usage
-----
Execute the script directly:

    python ternary_plot_example.py

A figure named ``ternary_predictions.png`` will be saved in the current
directory.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ternary

# Seed the random generator for reproducibility
random.seed(42)
np.random.seed(42)


def random_simplex_point() -> Tuple[float, float, float]:
    """Draw a random point on the 2-simplex (a+b+c = 1, a,b,c >= 0)."""

    # Sample from the Dirichlet distribution to obtain barycentric coordinates
    a, b, c = np.random.dirichlet(alpha=[1, 1, 1])
    return float(a), float(b), float(c)


def synthetic_prediction(a: float, b: float, c: float) -> float:
    """Toy model prediction as a function of mixture composition.

    This function mimics a response surface where the output depends on the
    product of the first two components and the proximity to the third
    component's vertex.
    """

    interaction = a * b
    vertex_bias = math.exp(-5 * (1 - c) ** 2)
    noise = np.random.normal(loc=0.0, scale=0.05)
    return interaction * vertex_bias + noise


def generate_dataset(n_samples: int = 120) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    """Generate synthetic ternary compositions and model predictions."""

    compositions: List[Tuple[float, float, float]] = []
    predictions: List[float] = []

    for _ in range(n_samples):
        composition = random_simplex_point()
        prediction = synthetic_prediction(*composition)
        compositions.append(composition)
        predictions.append(prediction)

    return compositions, predictions


def normalize_predictions(predictions: List[float]) -> List[float]:
    """Scale predictions to the [0, 1] interval for color mapping."""

    min_pred = min(predictions)
    max_pred = max(predictions)
    scale = max_pred - min_pred or 1.0
    return [(value - min_pred) / scale for value in predictions]


def plot_ternary_scatter(
    compositions: List[Tuple[float, float, float]],
    predictions: List[float],
    output_path: Path,
) -> None:
    """Create and save a ternary scatter plot."""

    scaled = normalize_predictions(predictions)
    cmap = plt.cm.viridis

    # Set up the ternary plot figure
    figure, tax = ternary.figure(scale=1.0)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=0.1, color="grey", linewidth=0.5, alpha=0.5)

    # Draw points
    tax.scatter(
        compositions,
        marker="o",
        color=[cmap(value) for value in scaled],
        s=40,
        edgecolors="black",
        linewidths=0.5,
    )

    # Configure ticks and labels with logarithmic-style values from 10^3 to 10^9
    exponents = range(3, 10)
    tick_positions = np.linspace(0.0, 1.0, num=len(exponents))
    tick_labels = [fr"$10^{{{exp}}}$" for exp in exponents]
    custom_ticks = list(zip(tick_positions, tick_labels))
    tax.ticks(axis="lbr", ticks=custom_ticks, linewidth=1, offset=0.02, fontsize=10)
    tax.left_axis_label("Molécula A", offset=0.12, fontsize=12)
    tax.right_axis_label("Molécula B", offset=0.12, fontsize=12)
    tax.top_axis_label("Molécula C", offset=0.12, fontsize=12)

    # Add a colorbar that maps predictions to colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(predictions), vmax=max(predictions)))
    sm.set_array([])
    cbar = figure.colorbar(sm, ax=tax.ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predicción del modelo", fontsize=12)

    tax.set_title("Predicciones de mezclas ternarias", fontsize=14)
    tax.clear_matplotlib_ticks()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    compositions, predictions = generate_dataset()
    output_path = Path("ternary_predictions.png")
    plot_ternary_scatter(compositions, predictions, output_path)
    print(f"Saved ternary plot to {output_path.resolve()}")


if __name__ == "__main__":
    main()
