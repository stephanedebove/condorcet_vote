# Author: Lê Nguyên Hoang
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def get_entities(df: pd.DataFrame, remove_nonevaluated: bool = True) -> List[str]:
    """
    Extracts entity/candidate names from DataFrame columns, optionally removing those not evaluated.
    Args:
        df (pd.DataFrame): DataFrame containing ballots, with columns for each entity.
        remove_nonevaluated (bool): If True, removes entities with no numeric evaluation.
    Returns:
        List[str]: List of entity/candidate names.
    Example:
        >>> get_entities(df)
        ['A', 'B', 'C']
    """
    print("Extracting entities from DataFrame columns.")
    return [
        e
        for e in df.columns[4:]
        if not remove_nonevaluated
        or any(pd.notnull(s) and str(s).replace(".", "", 1).isdigit() for s in df[e])
    ]


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
            if pd.notnull(row[e]) and str(row[e]).replace(".", "", 1).isdigit()
        ]
        for i_index, (i, score_i) in enumerate(scores):
            for j_index in range(i_index, len(scores)):
                j, score_j = scores[j_index]
                m[i, j] += np.sign(score_i - score_j)
                m[j, i] -= np.sign(score_i - score_j)
    return m


def compute_condorcet_lottery(
    comparison_matrix: np.ndarray, n_samples: Optional[int] = None
) -> np.ndarray:
    """
    Computes the Condorcet lottery (probability distribution) over candidates using linear programming.
    If n_samples is provided, averages over multiple runs for stability.
    Args:
        comparison_matrix (np.ndarray): Pairwise comparison matrix from get_comparison_matrix.
        n_samples (Optional[int]): Number of times to run the lottery and average results.
    Returns:
        np.ndarray: Probability for each entity to win (Condorcet lottery).
    Example:
        >>> compute_condorcet_lottery(matrix, n_samples=10)
        array([0.5, 0.5])
    """
    if n_samples:
        print(f"Computing Condorcet lottery with {n_samples} samples of averaging.")
    else:
        print("Computing Condorcet lottery without averaging.")
    n = len(comparison_matrix)
    kwargs = dict(
        A_ub=-np.sign(comparison_matrix).T,
        b_ub=np.zeros(n),
        A_eq=np.ones((1, n)),
        b_eq=np.ones(1),
    )
    if n_samples is None:
        return linprog(c=np.random.normal(0, 1, n), **kwargs).x
    else:

        def rsolve() -> np.ndarray:
            return linprog(c=np.random.normal(0, 1, n), **kwargs).x

        return np.array([rsolve() for _ in range(n_samples)]).mean(axis=0)


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
    Example:
        >>> sample_lottery(['A', 'B'], np.array([0.5, 0.5]), n_samples=1)
        array(['A'], dtype='<U1')
    """
    print(f"Sampling {n_samples} winner(s) from Condorcet lottery.")
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

    parser = argparse.ArgumentParser(
        description="Analyze Condorcet voting results from a CSV file."
    )
    parser.add_argument(
        "csv_path", type=str, help="Path to the CSV file with voting results."
    )
    parser.add_argument(
        "--full-lottery",
        action="store_true",
        help="Print the full Condorcet lottery (probability distribution) for all candidates instead of a single winner.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of winners to sample if not printing the full lottery. Default is number of candidates.",
    )
    parser.add_argument(
        "--lottery-averaging",
        type=int,
        default=None,
        help="Number of times to average the Condorcet lottery computation for stability. Default is number of candidates.",
    )
    parser.add_argument(
        "--remove-nonevaluated",
        action="store_true",
        help="Remove entities that have not been evaluated by any voter.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    entities = get_entities(df, remove_nonevaluated=args.remove_nonevaluated)
    comparison_matrix = get_comparison_matrix(entities, df)
    n_candidates = len(entities)
    n_samples = args.n_samples if args.n_samples is not None else n_candidates
    lottery_averaging = (
        args.lottery_averaging if args.lottery_averaging is not None else n_candidates
    )
    condorcet_lottery = compute_condorcet_lottery(
        comparison_matrix, n_samples=lottery_averaging
    )

    if args.full_lottery:
        print("Condorcet lottery (probability distribution) for each candidate:")
        for entity, prob in zip(entities, condorcet_lottery):
            print(f"  {entity}: {prob:.4f}")
    else:
        winners = sample_lottery(entities, condorcet_lottery, n_samples=n_samples)
        if n_samples == 1:
            print(f"The randomly selected Condorcet winner is: {winners[0]}")
        else:
            print(f"The randomly selected Condorcet winners are: {', '.join(winners)}")
