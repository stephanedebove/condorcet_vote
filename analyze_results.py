# Author: Lê Nguyên Hoang
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def get_comparison_matrix(entities: list, df: pd.DataFrame) -> np.ndarray:
    """
    For each pair of candidates, counts how many times one is preferred over the other across all ballots.
    Args:
        entities (list): List of candidate/entity names.
        df (pd.DataFrame): DataFrame containing ballots, with columns for each entity.
    Returns:
        np.ndarray: Square matrix where entry [i, j] is positive if entity i is preferred over j more often.
    """
    m = np.zeros((len(entities), len(entities)))
    for _, row in df.iterrows():
        scores = [
            (i, float(row[e]))
            for i, e in enumerate(entities)
            if pd.notnull(row[e]) and str(row[e]).replace('.', '', 1).isdigit()
        ]
        for i_index, (i, score_i) in enumerate(scores):
            for j_index in range(i_index, len(scores)):
                j, score_j = scores[j_index]
                m[i, j] += np.sign(score_i - score_j)
                m[j, i] -= np.sign(score_i - score_j)
    return m


def compute_condorcet_lottery(comparison_matrix: np.ndarray) -> np.ndarray:
    """
    If there is no single Condorcet winner, this function uses linear programming to find a fair probability distribution (lottery) over the candidates, based on the pairwise preferences.
    Args:
        comparison_matrix (np.ndarray): Pairwise comparison matrix from get_comparison_matrix.
    Returns:
        np.ndarray: Probability for each entity to win (Condorcet lottery).
    """
    return linprog(
        c=np.random.normal(0, 1, len(comparison_matrix)),
        A_ub=-np.sign(comparison_matrix).T,
        b_ub=np.zeros(len(comparison_matrix)),
        A_eq=np.ones((1, len(comparison_matrix))),
        b_eq=np.ones(1),
    ).x


def sample_lottery(
    entities: list, lottery: np.ndarray, n_samples: int = 1
) -> np.ndarray:
    """
    Randomly selects a winner (or winners) according to the computed probabilities.
    Args:
        entities (list): List of candidate/entity names.
        lottery (np.ndarray): Probability distribution over entities.
        n_samples (int): Number of samples to draw.
    Returns:
        np.ndarray: Sampled entity/entities.
    """
    return np.random.choice(entities, n_samples, p=lottery)


def _test():
    dominance_matrix = np.array(
        [[0, 1, 1, 1], [-1, 0, 1, -1], [-1, -1, 0, 1], [-1, 1, -1, 0]]
    )
    dominance_lottery = compute_condorcet_lottery(dominance_matrix)
    assert dominance_lottery[0] > 0.99
    assert all({dominance_lottery[i] < 0.01 for i in range(1, 4)})

    chifoumi_matrix = np.array(
        [[0, 1, -1, 1], [-1, 0, 1, -1], [1, -1, 0, 1], [-1, 1, -1, 0]]
    )
    chifoumi_lottery = compute_condorcet_lottery(chifoumi_matrix)
    assert all(
        {chifoumi_lottery[i] > 0.33 and chifoumi_lottery[i] < 0.34 for i in range(3)}
    )
    assert chifoumi_lottery[3] < 0.01


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Condorcet voting results from a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file with voting results.")
    parser.add_argument(
        "--full-lottery",
        action="store_true",
        help="Print the full Condorcet lottery (probability distribution) for all candidates instead of a single winner."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of winners to sample if not printing the full lottery. Default is 1."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    entities = list(df.columns)[4:]
    comparison_matrix = get_comparison_matrix(entities, df)
    condorcet_lottery = compute_condorcet_lottery(comparison_matrix)

    if args.full_lottery:
        print("Condorcet lottery (probability distribution) for each candidate:")
        for entity, prob in zip(entities, condorcet_lottery):
            print(f"  {entity}: {prob:.4f}")
    else:
        winners = sample_lottery(entities, condorcet_lottery, n_samples=args.n_samples)
        if args.n_samples == 1:
            print(f"The randomly selected Condorcet winner is: {winners[0]}")
        else:
            print(f"The randomly selected Condorcet winners are: {', '.join(winners)}")
