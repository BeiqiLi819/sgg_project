import torch

from sgg.evaluation.sgg_eval import (
    _degree_gini,
    _label_pair_entropy,
    _pressure_bucket,
)


def test_pressure_buckets_use_fixed_star_top10000_budget():
    assert _pressure_bucket(10000) == "no_truncation"
    assert _pressure_bucket(10001) == "overload_low"
    assert _pressure_bucket(20000) == "overload_low"
    assert _pressure_bucket(20001) == "overload_medium"
    assert _pressure_bucket(50000) == "overload_medium"
    assert _pressure_bucket(50001) == "overload_high"


def test_degree_gini_distinguishes_balanced_and_hub_graphs():
    assert _degree_gini(torch.tensor([2.0, 2.0, 2.0])) == 0.0
    assert _degree_gini(torch.tensor([6.0, 0.0, 0.0])) > 0.6


def test_label_pair_entropy_is_zero_for_one_type_and_positive_for_two():
    labels = torch.tensor([1, 1, 2])
    assert _label_pair_entropy(labels, torch.tensor([[0, 1]])) == 0.0
    assert _label_pair_entropy(labels, torch.tensor([[0, 1], [0, 2]])) > 0.0
