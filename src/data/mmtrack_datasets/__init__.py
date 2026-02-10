"""Compatibility re-exports for moved legacy MMTrack dataset classes."""

from legacy.mmtrack_datasets import (  # noqa: F401
    BDD100KDetDataset,
    BDDVideoDataset,
    ProbabilisticCocoDataset,
    ProbabilisticCocoVideoDataset,
    ProbabilisticMOTChallengeDataset,
)

__all__ = [
    "ProbabilisticCocoDataset",
    "ProbabilisticCocoVideoDataset",
    "BDDVideoDataset",
    "BDD100KDetDataset",
    "ProbabilisticMOTChallengeDataset",
]

