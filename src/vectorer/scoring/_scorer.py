"""FellegiSunterScorer: the assembled scoring class.

The class is composed from five focused mixins so each concern stays readable:

* :class:`~vectorer.scoring._construction.ConstructionMixin` -- constructor,
  ``from_comparisons`` / ``from_settings`` / save / load;
* :class:`~vectorer.scoring._inference.InferenceMixin` -- scalar evaluation
  and the public scoring API;
* :class:`~vectorer.scoring._union.UnionClassMixin` -- the Union-Class
  existential lift;
* :class:`~vectorer.scoring._training.TrainingMixin` -- ``calibrate_from_pairs``,
  ``fit_em`` and prior recovery;
* :class:`~vectorer.scoring._candidates.CandidatesMixin` -- candidate-pair
  generation entry points.
"""

from __future__ import annotations

from ._candidates import CandidatesMixin
from ._construction import ConstructionMixin
from ._inference import InferenceMixin
from ._training import TrainingMixin
from ._union import UnionClassMixin


class FellegiSunterScorer(
    ConstructionMixin,
    InferenceMixin,
    UnionClassMixin,
    TrainingMixin,
    CandidatesMixin,
):
    """Scores pairs by applying (trained) ``m/u`` per comparison level.

    Level assignment uses an ordered priority (first matching level wins), but
    evaluated as vectorized NumPy predicates over a whole batch of pairs.  Two
    pair inputs are supported:

    * :meth:`score_batch` / :meth:`match_weight_batch` - one query record vs a
      list of candidate records (the incremental pipeline);
    * :meth:`score_pairs` / :meth:`match_weight_pairs` - equal-length left /
      right record lists (the batch pipeline's canopy pairs).
    """