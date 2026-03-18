import copy
import logging
import resource
import time
from dataclasses import dataclass
from collections import defaultdict, namedtuple
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
from heapq import heappush, heappushpop, nlargest

import numpy as np
from alibi.utils.distributions import kl_bernoulli

logger = logging.getLogger(__name__)


@dataclass
class RuleBatchStats:
    """Minimal stats needed to compute rule interestingness."""

    p_a: np.ndarray  # P(A): coverage for each anchor
    p_b_given_a: np.ndarray  # P(B|A): precision/mean for each anchor
    p_b: float  # P(B): base rate for the target label y
    lb_prec: np.ndarray  # lower bound on precision
    ub_prec: np.ndarray  # upper bound on precision
    n: np.ndarray  # samples used to estimate p_b_given_a


def obj_wracc(stats: RuleBatchStats) -> np.ndarray:
    # WRAcc = P(A) * (P(B|A) - P(B)) == P(AB) - P(A)P(B)  (Hamilton survey)
    return stats.p_a * (stats.p_b_given_a - stats.p_b)


def obj_leverage(stats: RuleBatchStats) -> np.ndarray:
    # Leverage = P(AB) - P(A)P(B)
    return stats.p_a * stats.p_b_given_a - stats.p_a * stats.p_b


def obj_lift(stats: RuleBatchStats) -> np.ndarray:
    # Lift = P(B|A) / P(B)
    denom = max(stats.p_b, 1e-12)
    return stats.p_b_given_a / denom


def obj_jaccard(stats: RuleBatchStats) -> np.ndarray:
    # Jaccard = P(AB) / (P(A)+P(B)-P(AB))
    p_ab = stats.p_a * stats.p_b_given_a
    denom = stats.p_a + stats.p_b - p_ab
    out = np.zeros_like(p_ab)
    mask = denom > 0
    out[mask] = p_ab[mask] / denom[mask]
    return out


def obj_coverage(stats: RuleBatchStats) -> np.ndarray:
    return stats.p_a


# Registry
OBJECTIVES: Dict[str, Callable[[RuleBatchStats], np.ndarray]] = {
    "wracc": obj_wracc,
    "leverage": obj_leverage,
    "lift": obj_lift,
    "jaccard": obj_jaccard,
    "coverage": obj_coverage,  # default (backward compatible)
}


# Built-in constraints (vectorized; return boolean mask)
def cons_lcb_precision(
    stats: RuleBatchStats, desired_conf: float, eps: float
) -> np.ndarray:
    # original behavior: means >= tau and LB > tau - eps
    means = stats.p_b_given_a
    return (means >= desired_conf) & (stats.lb_prec > (desired_conf - eps))


def cons_effect_support(
    stats: RuleBatchStats, min_support: float = 0.01, min_lift: float = 1.5
) -> np.ndarray:
    return (stats.p_a >= min_support) & (obj_lift(stats) >= min_lift)


def cons_min_leverage(
    stats: RuleBatchStats, min_lev: float = 0.001, min_support: float = 0.0
) -> np.ndarray:
    return (obj_leverage(stats) >= min_lev) & (stats.p_a >= min_support)


CONSTRAINTS: Dict[str, Callable[..., np.ndarray]] = {
    "lcb_precision": cons_lcb_precision,  # default (backward compatible)
    "effect_support": cons_effect_support,
    "min_leverage": cons_min_leverage,
}


# TODO: Discuss logging strategy


class AnchorBaseBeam:

    def __init__(self, samplers: List[Callable], **kwargs) -> None:
        """
        Parameters
        ---------
        samplers
            Objects that can be called with args (`result`, `n_samples`) tuple to draw samples.
        """

        self.sample_fcn = samplers[0]
        self.samplers: Optional[List[Callable]] = samplers
        # Initial size (in batches) of data/raw data samples cache.
        self.sample_cache_size = kwargs.get("sample_cache_size", 10000)
        # when only the max of self.margin or batch size remain emptpy, the cache is
        # extended to accommodate an additional sample_cache_size batches.
        self.margin = kwargs.get("cache_margin", 1000)
        self.max_perturbation_batch_size = int(
            kwargs.get("max_perturbation_batch_size", 0) or 0
        )
        self.stats_provider = kwargs.get("stats_provider", None)
        self.instrumentation: Dict[str, Union[int, float, bool]] = {
            "start_time": 0.0,
            "end_time": 0.0,
            "time_per_explanation_s": 0.0,
            "peak_rss_mb": 0.0,
            "model_calls": 0,
            "perturbation_samples_evaluated": 0,
            "predictor_invocations": 0,
            "predicted_samples_total": 0,
            "sample_calls": 0,
            "samples_drawn": 0,
            "cached_state_data_bytes": 0,
            "cached_state_labels_bytes": 0,
            "cached_state_shapes": {},
        }

    def _start_instrumentation(self) -> None:
        self.instrumentation["start_time"] = time.perf_counter()

    def _finalize_instrumentation(self) -> None:
        end = time.perf_counter()
        self.instrumentation["end_time"] = end
        self.instrumentation["time_per_explanation_s"] = end - float(
            self.instrumentation["start_time"]
        )
        self.instrumentation["peak_rss_mb"] = (
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        )
        self._refresh_sampler_counters()
        data_arr = self.state.get("data") if hasattr(self, "state") else None
        labels_arr = self.state.get("labels") if hasattr(self, "state") else None
        data_bytes = int(getattr(data_arr, "nbytes", 0) or 0)
        labels_bytes = int(getattr(labels_arr, "nbytes", 0) or 0)
        self.instrumentation["cached_state_data_bytes"] = data_bytes
        self.instrumentation["cached_state_labels_bytes"] = labels_bytes
        self.instrumentation["cached_state_shapes"] = {
            "data": tuple(getattr(data_arr, "shape", ())),
            "labels": tuple(getattr(labels_arr, "shape", ())),
        }

    def _refresh_sampler_counters(self) -> None:
        model_calls = 0
        perturbation_samples = 0
        providers: List = []
        if self.stats_provider is not None:
            providers.append(self.stats_provider)
        providers.extend(self.samplers or [self.sample_fcn])
        for sampler in providers:
            model_calls += int(getattr(sampler, "model_calls", 0))
            perturbation_samples += int(
                getattr(sampler, "perturbation_samples_evaluated", 0)
            )
        self.instrumentation["model_calls"] = model_calls
        self.instrumentation["perturbation_samples_evaluated"] = perturbation_samples
        self.instrumentation["predictor_invocations"] = model_calls
        self.instrumentation["predicted_samples_total"] = perturbation_samples

    def _init_state(self, batch_size: int, coverage_data: np.ndarray) -> None:
        """
        Initialises the object state, which is used to compute result precisions & precision bounds
        and provide metadata for explanation objects.

        Parameters
        ----------
        batch_size
            See :py:meth:`alibi.explainers.anchors.anchor_base.AnchorBaseBeam.anchor_beam` method.
        coverage_data
            See :py:meth:`alibi.explainers.anchors.anchor_base.AnchorBaseBeam._get_coverage_samples` method.
        """

        prealloc_size = batch_size * self.sample_cache_size
        # t_ indicates that the attribute is a dictionary with entries for each anchor
        self.state: dict = {
            "t_coverage": defaultdict(lambda: 0.0),  # anchors' coverage
            "t_coverage_idx": defaultdict(set),  # index of anchors in coverage set
            "t_covered_true": defaultdict(
                None
            ),  # samples with same pred as instance where t_ applies
            "t_covered_false": defaultdict(
                None
            ),  # samples with dif pred to instance where t_ applies
            "t_idx": defaultdict(
                set
            ),  # row idx in sample cache where the anchors apply
            "t_nsamples": defaultdict(
                lambda: 0.0
            ),  # total number of samples drawn for the anchors
            "t_order": defaultdict(
                list
            ),  # anchors are sorted to avoid exploring permutations
            # this is the order in which anchors were found
            "t_positives": defaultdict(
                lambda: 0.0
            ),  # nb of samples where result pred = pred on instance
            "prealloc_size": prealloc_size,  # samples caches size
            "data": np.zeros(
                (prealloc_size, coverage_data.shape[1]), coverage_data.dtype
            ),  # samples caches
            "labels": np.zeros(
                prealloc_size,
            ),  # clf pred labels on raw_data
            "current_idx": 0,
            "n_features": coverage_data.shape[1],  # data set dim after encoding
            "coverage_data": coverage_data,  # coverage data
        }
        self.state["t_order"][()] = ()  # Trivial order for the empty result

    @staticmethod
    def _sort(x: tuple, allow_duplicates=False) -> tuple:
        """
        Sorts a tuple, optionally removing duplicates.

        Parameters
        ----------
        x:
            Tuple to be sorted.
        allow_duplicates:
            If ``True``, duplicate entries are kept.

        Returns
        -------
        A sorted tuple.
        """

        if allow_duplicates:
            return tuple(sorted(x))

        return tuple(sorted(set(x)))

    @staticmethod
    def dup_bernoulli(p: np.ndarray, level: np.ndarray, n_iter: int = 17) -> np.ndarray:
        """
        Update upper precision bound for a candidate anchors dependent on the KL-divergence.

        Parameters
        ----------
        p
            Precision of candidate anchors.
        level
            `beta / nb of samples` for each result.
        n_iter
            Number of iterations during lower bound update.

        Returns
        -------
        Updated upper precision bounds array.
        """
        # TODO: where does 17x sampling come from?
        lm = p.copy()
        um = np.minimum(np.minimum(p + np.sqrt(level / 2.0), 1.0), 1.0)

        # Perform bisection algorithm to find the largest qm s.t. kl divergence is > level
        for j in range(1, n_iter):
            qm = (um + lm) / 2.0
            kl_gt_idx = kl_bernoulli(p, qm) > level
            kl_lt_idx = np.logical_not(kl_gt_idx)
            um[kl_gt_idx] = qm[kl_gt_idx]
            lm[kl_lt_idx] = qm[kl_lt_idx]

        return um

    @staticmethod
    def dlow_bernoulli(
        p: np.ndarray, level: np.ndarray, n_iter: int = 17
    ) -> np.ndarray:
        """
        Update lower precision bound for a candidate anchors dependent on the KL-divergence.

        Parameters
        ----------
        p
            Precision of candidate anchors.
        level
            `beta / nb of samples` for each result.
        n_iter
            Number of iterations during lower bound update.

        Returns
        -------
        Updated lower precision bounds array.
        """

        um = p.copy()
        lm = np.clip(p - np.sqrt(level / 2.0), 0.0, 1.0)  # lower bound

        # Perform bisection algorithm to find the smallest qm s.t. kl divergence is > level
        for _ in range(1, n_iter):
            qm = (um + lm) / 2.0
            kl_gt_idx = kl_bernoulli(p, qm) > level
            kl_lt_idx = np.logical_not(kl_gt_idx)
            lm[kl_gt_idx] = qm[kl_gt_idx]
            um[kl_lt_idx] = qm[kl_lt_idx]

        return lm

    @staticmethod
    def compute_beta(n_features: int, t: int, delta: float) -> float:
        """
        Parameters
        ----------
        n_features
            Number of candidate anchors.
        t
            Iteration number.
        delta
            Confidence budget, candidate anchors have close to optimal precisions with prob. `1 - delta`.

        Returns
        -------
        Level used to update upper and lower precision bounds.
        """
        # The following constants are defined and used in the paper introducing the KL-LUCB bandit algorithm
        # (http://proceedings.mlr.press/v30/Kaufmann13.html). Specifically, Theorem 1 proves the lower bounds
        # of these constants to ensure the algorithm is PAC with probability at least 1-delta. Also refer to
        # section "5. Numerical experiments" where these values are used empirically.
        alpha = 1.1
        k = 405.5
        temp = np.log(k * n_features * (t**alpha) / delta)

        return temp + np.log(temp)

    def _get_coverage_samples(
        self, coverage_samples: int, samplers: Optional[List[Callable]] = None
    ) -> np.ndarray:
        """
        Draws samples uniformly at random from the training set.

        Parameters
        ---------
        coverage_samples
            See :py:meth:`alibi.explainers.anchors.anchor_base.AnchorBaseBeam.anchor_beam` method.
        samplers
            See :py:meth:`alibi.explainers.anchors.anchor_base.AnchorBaseBeam.__init__` method.

        Returns
        -------
        coverage_data
            Binarised samples, where 1 indicates the feature has same value/is in same beam as
            instance to be explained. Used to determine, e.g., which samples an result applies to.
        """

        [coverage_data] = self.sample_fcn(
            (0, ()), coverage_samples, compute_labels=False
        )

        return coverage_data

    def select_critical_arms(
        self,
        means: np.ndarray,
        ub: np.ndarray,
        lb: np.ndarray,
        n_samples: np.ndarray,
        delta: float,
        top_n: int,
        t: int,
    ):
        """
        Determines a set of two anchors by updating the upper bound for low empirical precision anchors and
        the lower bound for anchors with high empirical precision.

        Parameters
        ----------
        means
            Empirical mean result precisions.
        ub
            Upper bound on result precisions.
        lb
            Lower bound on result precisions.
        n_samples
            The number of samples drawn for each candidate result.
        delta
            Confidence budget, candidate anchors have close to optimal precisions with prob. `1 - delta`.
        top_n
            Number of arms to be selected.
        t
            Iteration number.

        Returns
        -------
        Upper and lower precision bound indices.
        """

        crit_arms = namedtuple("crit_arms", ["ut", "lt"])

        sorted_means = np.argsort(
            means
        )  # ascending sort of result candidates by precision
        beta = self.compute_beta(len(means), t, delta)

        # J = the beam width top result candidates with highest precision
        # not_J = the rest
        J = sorted_means[-top_n:]
        not_J = sorted_means[:-top_n]

        # update upper bound for lowest precision result candidates
        ub[not_J] = self.dup_bernoulli(means[not_J], beta / n_samples[not_J])
        # update lower bound for highest precision result candidates
        lb[J] = self.dlow_bernoulli(means[J], beta / n_samples[J])

        # for the low precision result candidates, compute the upper precision bound and keep the index ...
        # ... of the result candidate with the highest upper precision value -> ut
        # for the high precision result candidates, compute the lower precision bound and keep the index ...
        # ... of the result candidate with the lowest lower precision value -> lt
        ut = not_J[np.argmax(ub[not_J])]
        lt = J[np.argmin(lb[J])]

        return crit_arms._make((ut, lt))

    def kllucb(
        self,
        anchors: list,
        init_stats: dict,
        epsilon: float,
        delta: float,
        batch_size: int,
        top_n: int,
        verbose: bool = False,
        verbose_every: int = 1,
    ) -> np.ndarray:
        """
        Implements the KL-LUCB algorithm (Kaufmann and Kalyanakrishnan, 2013).

        Parameters
        ----------
        anchors:
            A list of anchors from which two critical anchors are selected (see Kaufmann and Kalyanakrishnan, 2013).
        init_stats
            Dictionary with lists containing nb of samples used and where sample predictions equal the desired label.
        epsilon
            Precision bound tolerance for convergence.
        delta
            Used to compute `beta`.
        batch_size
            Number of samples.
        top_n
            Min of beam width size or number of candidate anchors.
        verbose
            Whether to print intermediate output.
        verbose_every
            Whether to print intermediate output every `verbose_every` steps.

        Returns
        -------
        Indices of best result options. Number of indices equals min of beam width or nb of candidate anchors.
        """

        # n_features equals to the nb of candidate anchors
        n_features = len(anchors)

        # arrays for total number of samples & positives (# samples where prediction equals desired label)
        n_samples, positives = init_stats["n_samples"], init_stats["positives"]
        anchors_to_sample, anchors_idx = [], []
        for f in np.where(n_samples == 0)[0]:
            anchors_to_sample.append(anchors[f])
            anchors_idx.append(f)

        if anchors_idx:
            pos, total = self.draw_samples(anchors_to_sample, 1)
            positives[anchors_idx] += pos
            n_samples[anchors_idx] += total

        if n_features == top_n:  # return all options b/c of beam search width
            return np.arange(n_features)

        # update the upper and lower precision bounds until the difference between the best upper ...
        # ... precision bound of the low precision anchors and the worst lower precision bound of the high ...
        # ... precision anchors is smaller than eps
        means = (
            positives / n_samples
        )  # fraction sample predictions equal to desired label
        ub, lb = np.zeros(n_samples.shape), np.zeros(n_samples.shape)
        t = 1
        crit_a_idx = self.select_critical_arms(
            means, ub, lb, n_samples, delta, top_n, t
        )
        B = ub[crit_a_idx.ut] - lb[crit_a_idx.lt]
        verbose_count = 0

        while B > epsilon:

            verbose_count += 1
            if verbose and verbose_count % verbose_every == 0:
                ut, lt = crit_a_idx
                print(
                    "Best: %d (mean:%.10f, n: %d, lb:%.4f)"
                    % (lt, means[lt], n_samples[lt], lb[lt]),
                    end=" ",
                )
                print(
                    "Worst: %d (mean:%.4f, n: %d, ub:%.4f)"
                    % (ut, means[ut], n_samples[ut], ub[ut]),
                    end=" ",
                )
                print("B = %.2f" % B)

            # draw samples for each critical result, update anchors' mean, upper and lower
            # bound precision estimate
            selected_anchors = [anchors[idx] for idx in crit_a_idx]
            pos, total = self.draw_samples(selected_anchors, batch_size)
            idx = list(crit_a_idx)
            positives[idx] += pos
            n_samples[idx] += total
            means = positives / n_samples
            t += 1
            crit_a_idx = self.select_critical_arms(
                means, ub, lb, n_samples, delta, top_n, t
            )
            B = ub[crit_a_idx.ut] - lb[crit_a_idx.lt]
        sorted_means = np.argsort(means)

        return sorted_means[-top_n:]

    def draw_samples(self, anchors: list, batch_size: int) -> Tuple[tuple, tuple]:
        """
        Parameters
        ----------
        anchors
            Anchors on which samples are conditioned.
        batch_size
            The number of samples drawn for each result.

        Returns
        -------
        A tuple of positive samples (for which prediction matches desired label) and a tuple of         total number of samples drawn.
        """

        for anchor in anchors:
            if anchor not in self.state["t_order"]:
                self.state["t_order"][anchor] = list(anchor)

        sample_stats: List[Tuple[int, int]] = []
        target_batch_size = batch_size

        for i, anchor in enumerate(anchors):
            anchor_pos = 0
            anchor_total = 0
            while anchor_total < target_batch_size:
                max_chunk = self.max_perturbation_batch_size or target_batch_size
                n_chunk = min(max_chunk, target_batch_size - anchor_total)
                samples = self.sample_fcn(
                    (i, tuple(self.state["t_order"][anchor])), num_samples=n_chunk
                )
                covered_true, covered_false, labels, *additionals, _ = samples
                p, t = self.update_state(
                    covered_true, covered_false, labels, additionals, anchor
                )
                anchor_pos += int(p)
                anchor_total += int(t)
                self.instrumentation["sample_calls"] += 1
                self.instrumentation["samples_drawn"] += int(t)
                self._refresh_sampler_counters()
            sample_stats.append((anchor_pos, anchor_total))

        pos, total = list(zip(*sample_stats)) if sample_stats else (tuple(), tuple())
        return pos, total

    def propose_anchors(self, previous_best: list) -> list:
        """
        Parameters
        ----------
        previous_best
            List with tuples of result candidates.

        Returns
        -------
        List with tuples of candidate anchors with additional metadata.
        """

        # compute some variables used later on
        state = self.state
        all_features = range(state["n_features"])
        coverage_data = state["coverage_data"]
        current_idx = state["current_idx"]
        data = state["data"][:current_idx]
        labels = state["labels"][:current_idx]

        # initially, every feature separately is an result
        if len(previous_best) == 0:
            tuples = [(x,) for x in all_features]
            for x in tuples:
                pres = data[:, x[0]].nonzero()[
                    0
                ]  # Select samples whose feat value is = to the result value
                state["t_idx"][x] = set(pres)
                state["t_nsamples"][x] = float(len(pres))
                state["t_positives"][x] = float(labels[pres].sum())
                state["t_order"][x].append(x[0])
                state["t_coverage_idx"][x] = set(coverage_data[:, x[0]].nonzero()[0])
                state["t_coverage"][x] = (
                    float(len(state["t_coverage_idx"][x])) / coverage_data.shape[0]
                )
            return tuples

        # create new anchors: add a feature to every result in current best
        new_tuples: Set[tuple] = set()
        for f in all_features:
            for t in previous_best:
                new_t = self._sort(t + (f,), allow_duplicates=False)
                if len(new_t) != len(t) + 1:  # Avoid repeating the same feature ...
                    continue
                if new_t not in new_tuples:
                    new_tuples.add(new_t)
                    state["t_order"][new_t] = copy.deepcopy(state["t_order"][t])
                    state["t_order"][new_t].append(f)
                    state["t_coverage_idx"][new_t] = state["t_coverage_idx"][
                        t
                    ].intersection(state["t_coverage_idx"][(f,)])
                    state["t_coverage"][new_t] = (
                        float(len(state["t_coverage_idx"][new_t]))
                        / coverage_data.shape[0]
                    )
                    t_idx = np.array(
                        list(state["t_idx"][t])
                    )  # indices of samples where the len-1 result applies
                    t_data = state["data"][t_idx]
                    present = np.where(t_data[:, f] == 1)[0]
                    state["t_idx"][new_t] = set(
                        t_idx[present]
                    )  # indices of samples where the proposed result applies
                    idx_list = list(state["t_idx"][new_t])
                    state["t_nsamples"][new_t] = float(len(idx_list))
                    state["t_positives"][new_t] = np.sum(state["labels"][idx_list])

        return list(new_tuples)

    def update_state(
        self,
        covered_true: np.ndarray,
        covered_false: np.ndarray,
        labels: np.ndarray,
        samples: Tuple[np.ndarray, float],
        anchor: tuple,
    ) -> Tuple[int, int]:
        """
        Updates the explainer state (see :py:meth:`alibi.explainers.anchors.anchor_base.AnchorBaseBeam.__init__`
        for full state definition).

        Parameters
        ----------
        covered_true
            Examples where the result applies and the prediction is the same as on
            the instance to be explained.
        covered_false
            Examples where the result applies and the prediction is the different to
            the instance to be explained.
        samples
            A tuple containing discretized data, coverage and the result sampled.
        labels
            An array indicating whether the prediction on the sample matches the label
            of the instance to be explained.
        anchor
            The result to be updated.

        Returns
        -------
        A tuple containing the number of instances equals desired label of observation \
        to be explained the total number of instances sampled, and the result that was sampled.
        """

        # data = binary matrix where 1 means a feature has the same value as the feature in the result
        data, coverage = samples
        n_samples = data.shape[0]

        current_idx = self.state["current_idx"]
        idxs = range(current_idx, current_idx + n_samples)
        self.state["t_idx"][anchor].update(idxs)
        self.state["t_nsamples"][anchor] += n_samples
        self.state["t_positives"][anchor] += labels.sum()
        self.state["t_covered_true"][anchor] = covered_true
        self.state["t_covered_false"][anchor] = covered_false
        self.state["data"][idxs] = data
        self.state["labels"][idxs] = labels
        self.state["current_idx"] += n_samples

        if self.state["current_idx"] >= self.state["data"].shape[0] - max(
            self.margin, n_samples
        ):
            prealloc_size = self.state["prealloc_size"]
            self.state["data"] = np.vstack(
                (
                    self.state["data"],
                    np.zeros((prealloc_size, data.shape[1]), data.dtype),
                )
            )
            self.state["labels"] = np.hstack(
                (self.state["labels"], np.zeros(prealloc_size, labels.dtype))
            )

        return labels.sum(), data.shape[0]

    def get_init_stats(self, anchors: list, coverages=False) -> dict:
        """
        Finds the number of samples already drawn for each result in anchors, their
        comparisons with the instance to be explained and, optionally, coverage.

        Parameters
        ----------
        anchors
            Candidate anchors.
        coverages
            If ``True``, the statistics returned contain the coverage of the specified anchors.

        Returns
        -------
        Dictionary with lists containing nb of samples used and where sample predictions equal the desired label.
        """

        def array_factory(size: tuple):
            return lambda: np.zeros(size)

        state = self.state
        stats: Dict[str, np.ndarray] = defaultdict(array_factory((len(anchors),)))
        for i, anchor in enumerate(anchors):
            stats["n_samples"][i] = state["t_nsamples"][anchor]
            stats["positives"][i] = state["t_positives"][anchor]
            if coverages:
                stats["coverages"][i] = state["t_coverage"][anchor]

        return stats

    def get_anchor_metadata(
        self, features: tuple, success, batch_size: int = 100
    ) -> dict:
        """
        Given the features contained in a result, it retrieves metadata such as the precision and
        coverage of the result and partial anchors and examples where the result/partial anchors
        apply and yield the same prediction as on the instance to be explained (`covered_true`)
        or a different prediction (`covered_false`).

        Parameters
        ----------
        features
            Sorted indices of features in result.
        success
            Indicates whether an anchor satisfying precision threshold was met or not.
        batch_size
            Number of samples among which positive and negative examples for partial anchors are
            selected if partial anchors have not already been explicitly sampled.

        Returns
        -------
        Anchor dictionary with result features and additional metadata.
        """

        state = self.state
        anchor: dict = {
            "feature": [],
            "mean": [],
            "precision": [],
            "coverage": [],
            "examples": [],
            "all_precision": 0,
            "num_preds": int(state["current_idx"]),
            "success": success,
        }
        current_t: tuple = tuple()
        # draw pos and negative example where partial result applies if not sampled during search
        to_resample, to_resample_idx = [], []
        for f in state["t_order"][features]:
            current_t = self._sort(current_t + (f,), allow_duplicates=False)

            # Metadata needs precision for each ordered prefix; ensure prefix is sampled first.
            if state["t_nsamples"][current_t] <= 0:
                state["t_order"][current_t] = list(current_t)
                self.draw_samples([current_t], batch_size)
                if state["t_nsamples"][current_t] <= 0:
                    raise RuntimeError(
                        f"Invariant violation: prefix {current_t} has zero samples after resampling."
                    )

            mean = state["t_positives"][current_t] / state["t_nsamples"][current_t]
            anchor["feature"].append(f)
            anchor["mean"].append(mean)
            anchor["precision"].append(mean)
            anchor["coverage"].append(state["t_coverage"][current_t])

            # add examples where result does or does not hold
            if current_t in state["t_covered_true"]:
                exs = {
                    "covered_true": state["t_covered_true"][current_t],
                    "covered_false": state["t_covered_false"][current_t],
                    "uncovered_true": np.array([]),
                    "uncovered_false": np.array([]),
                }
                anchor["examples"].append(exs)
            else:
                to_resample.append(current_t)
                # sampling process relies on ordering
                state["t_order"][current_t] = list(current_t)
                to_resample_idx.append(len(anchor["examples"]))
                anchor["examples"].append("placeholder")
                # if the anchor was not sampled, the coverage is not estimated
                anchor["coverage"][-1] = "placeholder"

        # If partial anchors have not been sampled, resample to find examples
        if to_resample:

            _, _ = self.draw_samples(to_resample, batch_size)

            while to_resample:
                feats, example_idx = to_resample.pop(), to_resample_idx.pop()
                anchor["examples"][example_idx] = {
                    "covered_true": state["t_covered_true"].get(feats, np.array([])),
                    "covered_false": state["t_covered_false"].get(feats, np.array([])),
                    "uncovered_true": np.array([]),
                    "uncovered_false": np.array([]),
                }
                # update result with true coverage
                anchor["coverage"][example_idx] = state["t_coverage"][feats]

        return anchor

    @staticmethod
    def to_sample(
        means: np.ndarray,
        ubs: np.ndarray,
        lbs: np.ndarray,
        desired_confidence: float,
        epsilon_stop: float,
    ):
        """
        Given an array of mean result precisions and their upper and lower bounds, determines for which anchors
        more samples need to be drawn in order to estimate the anchors precision with `desired_confidence` and error
        tolerance.

        Parameters
        ----------
        means:
            Mean precisions (each element represents a different result).
        ubs:
            Precisions' upper bounds (each element represents a different result).
        lbs:
            Precisions' lower bounds (each element represents a different result).
        desired_confidence:
            Desired level of confidence for precision estimation.
        epsilon_stop:
            Tolerance around desired precision.

        Returns
        -------
        Boolean array indicating whether more samples are to be drawn for that particular result.
        """

        return (
            (means >= desired_confidence) & (lbs < desired_confidence - epsilon_stop)
        ) | ((means < desired_confidence) & (ubs >= desired_confidence + epsilon_stop))

    def anchor_beam(
        self,
        delta: float = 0.05,
        epsilon: float = 0.1,
        desired_confidence: float = 1.0,
        beam_size: int = 1,
        epsilon_stop: float = 0.05,
        min_samples_start: int = 100,
        max_anchor_size: Optional[int] = None,
        stop_on_first: bool = False,
        batch_size: int = 100,
        coverage_samples: int = 10000,
        verbose: bool = False,
        verbose_every: int = 1,
        objective: Union[str, Callable[[RuleBatchStats], np.ndarray]] = "coverage",
        constraint: Union[str, Callable[..., np.ndarray]] = "lcb_precision",
        constraint_kwargs: Optional[dict] = None,
        top_k_return: int = 1,
        **kwargs,
    ) -> dict:
        """
        Uses the KL-LUCB algorithm (Kaufmann and Kalyanakrishnan, 2013) together with additional sampling to search
        feature sets (anchors) that guarantee the prediction made by a classifier model. The search is greedy if
        ``beam_size=1``. Otherwise, at each of the `max_anchor_size` steps, `beam_size` solutions are explored.
        By construction, solutions found have high precision (defined as the expected of number of times the classifier
        makes the same prediction when queried with the feature subset combined with arbitrary samples drawn from a
        noise distribution). The algorithm maximises the coverage of the solution found - the frequency of occurrence
        of records containing the feature subset in set of samples.

        Optimize an arbitrary objective M(A→y) under a reliability constraint (default = LCB-precision).
        Defaults preserve original behavior: maximize coverage subject to precision (Anchors paper).


        Parameters
        ----------
        delta
            Used to compute `beta`.
        epsilon
            Precision bound tolerance for convergence.
        desired_confidence
            Desired level of precision (`tau` in `paper <https://homes.cs.washington.edu/~marcotcr/aaai18.pdf>`_).
        beam_size
            Beam width.
        epsilon_stop
            Confidence bound margin around desired precision.
        min_samples_start
            Min number of initial samples.
        max_anchor_size
            Max number of features in result.
        stop_on_first
            Stop on first valid result found.
        coverage_samples
            Number of samples from which to build a coverage set.
        batch_size
            Number of samples used for an arm evaluation.
        verbose
            Whether to print intermediate LUCB & anchor selection output.
        verbose_every
            Print intermediate output every verbose_every steps.

        Returns
        -------
        Explanation dictionary containing anchors with metadata like coverage and precision and examples.
        """

        if constraint_kwargs is None:
            constraint_kwargs = {}
        self._start_instrumentation()

        objective_fn = (
            OBJECTIVES[objective] if isinstance(objective, str) else objective
        )
        constraint_fn = (
            CONSTRAINTS[constraint] if isinstance(constraint, str) else constraint
        )

        is_precision_constraint = (
            isinstance(constraint, str) and constraint == "lcb_precision"
        ) or (constraint_fn is cons_lcb_precision)
        constraint_label = (
            constraint if isinstance(constraint, str) else constraint_fn.__name__
        )
        objective_label = (
            objective if isinstance(objective, str) else objective.__name__
        )

        # Select coverage set and initialise object state
        coverage_data = self._get_coverage_samples(
            coverage_samples,
            samplers=self.samplers,
        )
        self._init_state(batch_size, coverage_data)

        # sample by default 1 or min_samples_start more random value(s)
        # sample by default 1 or min_samples_start more random value(s)
        (pos,), (total,) = self.draw_samples([()], min_samples_start)

        mean = np.array([pos / total])  # P(B) and P(B|A=∅)
        beta = np.log(1.0 / delta)
        lb = self.dlow_bernoulli(mean, np.array([beta / total]))
        ub = lb  # not used here, but keep shape if you like

        # Evaluate constraint on the empty anchor
        if is_precision_constraint:
            ok_empty = cons_lcb_precision(
                RuleBatchStats(
                    p_a=np.array([1.0]),  # A=∅ holds everywhere
                    p_b_given_a=mean,  # precision on empty rule
                    p_b=float(mean),  # base rate
                    lb_prec=lb,
                    ub_prec=lb,
                    n=np.array([total]),
                ),
                desired_confidence,
                epsilon_stop,
            )
        else:
            ok_empty = constraint_fn(
                RuleBatchStats(
                    p_a=np.array([1.0]),
                    p_b_given_a=mean,
                    p_b=float(mean),
                    lb_prec=lb,
                    ub_prec=lb,
                    n=np.array([total]),
                ),
                **(constraint_kwargs or {}),
            )

        if np.asarray(ok_empty).item():
            result = {
                "feature": [],
                "mean": [],
                "num_preds": total,
                "precision": [],
                "coverage": [1.0],
                "examples": [],
                "all_precision": mean,
                "success": True,
                "objective_name": objective_label,
                "constraint_name": constraint_label,
                "score": [
                    float(
                        objective_fn(
                            RuleBatchStats(
                                p_a=np.array([1.0]),
                                p_b_given_a=mean,
                                p_b=float(mean),
                                lb_prec=lb,
                                ub_prec=lb,
                                n=np.array([total]),
                            )
                        )[0]
                    )
                ],
            }
            self._finalize_instrumentation()
            result["instrumentation"] = copy.deepcopy(self.instrumentation)
            return result

        current_size = 1
        best_score = -np.inf
        best_of_size: Dict[int, list] = {0: []}
        best_anchor = ()
        best_payload = None
        topk_heap = []
        heap_push_counter = (
            0  # deterministic tie-breaker to avoid dict comparison when scores tie
        )
        seen = set()  # avoid duplicates across sizes

        if max_anchor_size is None:
            max_anchor_size = self.state["n_features"]

        # find best result using beam search
        while current_size <= max_anchor_size:

            # create new candidate anchors by adding features to current best anchors
            anchors = self.propose_anchors(best_of_size[current_size - 1])

            # if no better coverage found with added features -> break
            if len(anchors) == 0:
                break

            # for each result, get initial nb of samples used and prec(A)
            stats = self.get_init_stats(anchors)

            # apply KL-LUCB and return result options (nb of options = beam width) in the form of indices
            candidate_anchors = self.kllucb(
                anchors,
                stats,
                epsilon,
                delta,
                batch_size,
                min(beam_size, len(anchors)),
                verbose=verbose,
                verbose_every=verbose_every,
            )
            # store best anchors for the given result size (nb of features in the result)
            best_of_size[current_size] = [anchors[index] for index in candidate_anchors]
            # for each candidate result:
            #   update precision, lower and upper bounds until precision constraints are met
            #   update best result if coverage is larger than current best coverage
            stats = self.get_init_stats(best_of_size[current_size], coverages=True)
            positives, n_samples = stats["positives"], stats["n_samples"]
            beta = np.log(
                1.0 / (delta / (1 + (beam_size - 1) * self.state["n_features"]))
            )
            kl_constraints = beta / n_samples
            means = stats["positives"] / stats["n_samples"]
            lbs = self.dlow_bernoulli(means, kl_constraints)
            ubs = self.dup_bernoulli(means, kl_constraints)

            if verbose:
                print("Best of size ", current_size, ":")
                for i, mean, lb, ub in zip(candidate_anchors, means, lbs, ubs):
                    print(i, mean, lb, ub)

            # draw samples to ensure result meets precision criteria
            continue_sampling = self.to_sample(
                means, ubs, lbs, desired_confidence, epsilon_stop
            )
            while continue_sampling.any():
                selected_anchors = [
                    anchors[idx] for idx in candidate_anchors[continue_sampling]
                ]
                pos, total = self.draw_samples(selected_anchors, batch_size)
                positives[continue_sampling] += pos
                n_samples[continue_sampling] += total
                means[continue_sampling] = (
                    positives[continue_sampling] / n_samples[continue_sampling]
                )
                kl_constraints[continue_sampling] = beta / n_samples[continue_sampling]
                lbs[continue_sampling] = self.dlow_bernoulli(
                    means[continue_sampling],
                    kl_constraints[continue_sampling],
                )
                ubs[continue_sampling] = self.dup_bernoulli(
                    means[continue_sampling],
                    kl_constraints[continue_sampling],
                )
                continue_sampling = self.to_sample(
                    means, ubs, lbs, desired_confidence, epsilon_stop
                )

            # anchors who meet the precision setting and have better coverage than the best anchors so far
            coverages = stats["coverages"]
            base_p = self.state.get("all_precision", None)
            if base_p is None:
                # fallback: estimate once by drawing with empty anchor
                (pos_b,), (tot_b,) = self.draw_samples([()], max(100, batch_size))
                base_p = float(pos_b) / float(tot_b)

            rb = RuleBatchStats(
                p_a=coverages,
                p_b_given_a=means,
                p_b=base_p,
                lb_prec=lbs,
                ub_prec=ubs,
                n=n_samples,
            )
            valid_mask = (
                constraint_fn(rb, desired_confidence, epsilon_stop, **constraint_kwargs)
                if constraint_fn is cons_lcb_precision
                else constraint_fn(rb, **constraint_kwargs)
            )
            scores = objective_fn(rb)

            # pick candidates with better score than best so far
            # log candidates of this size (optional)
            if verbose:
                print("Best of size ", current_size, ":")
                for i in range(len(best_of_size[current_size])):
                    t = best_of_size[current_size][i]
                    print(
                        f"{t} mean={means[i]:.3f} lb={lbs[i]:.3f} ub={ubs[i]:.3f} P(A)={coverages[i]:.3f} "
                        f'score={scores[i]:.4f} valid={bool(valid_mask[i])} by constraint="{constraint_label}" score={scores[i]:.4f}'
                    )

            # --- collect top-K valid anchors by objective ---
            for idx in range(len(best_of_size[current_size])):
                if not valid_mask[idx]:
                    continue
                a = best_of_size[current_size][idx]
                if a in seen:
                    continue
                seen.add(a)

                payload = dict(
                    anchor=a,
                    score=float(scores[idx]),
                    score_name=(
                        objective if isinstance(objective, str) else objective.__name__
                    ),
                    constraint_name=(
                        constraint
                        if isinstance(constraint, str)
                        else constraint.__name__
                    ),
                    p_a=float(coverages[idx]),
                    p_b=float(base_p),
                    p_b_given_a=float(means[idx]),
                    lb=float(lbs[idx]),
                    ub=float(ubs[idx]),
                    n=int(n_samples[idx]),
                )

                # maintain a min-heap of size <= top_k_return
                if top_k_return > 0:
                    entry = (payload["score"], heap_push_counter, payload)
                    heap_push_counter += 1
                    if len(topk_heap) < top_k_return:
                        heappush(topk_heap, entry)
                    elif payload["score"] > topk_heap[0][0]:
                        heappushpop(topk_heap, entry)

                # keep backward-compatible "single best" selection
                if payload["score"] > best_score:
                    best_score = payload["score"]
                    best_anchor = a
                    best_payload = payload
                    if stop_on_first:
                        break

            current_size += 1

        # if no result is found, choose the highest-OBJECTIVE candidate from all rounds (ignoring the constraint)

        if not best_anchor:
            success = False  # indicates the method has not found an anchor
            logger.warning(
                f'No anchor satisfied the "{constraint_label}" constraint. '
                f'Returning the best candidate by objective "{objective_label}" without enforcing the constraint.'
                + (
                    f" (requested precision threshold={desired_confidence})"
                    if is_precision_constraint
                    else ""
                )
            )
            anchors = []
            for i in range(0, current_size):
                anchors.extend(best_of_size[i])
            if len(anchors) == 0:
                result = {
                    "feature": [],
                    "mean": [],
                    "num_preds": int(total),
                    "precision": [],
                    "coverage": [],
                    "examples": [],
                    "all_precision": mean,
                    "success": False,
                    "objective_name": objective_label,
                    "constraint_name": constraint_label,
                }
                self._finalize_instrumentation()
                result["instrumentation"] = copy.deepcopy(self.instrumentation)
                return result

            # score all anchors by the chosen objective (ignore constraint)
            stats = self.get_init_stats(anchors, coverages=True)
            positives, n_samples = stats["positives"], stats["n_samples"]
            means = positives / n_samples
            coverages = stats["coverages"]

            # bounds (not strictly needed for objective, but may be useful downstream)
            beta_all = np.log(
                1.0 / (delta / (1 + (beam_size - 1) * self.state["n_features"]))
            )
            kl_all = beta_all / n_samples
            lbs_all = self.dlow_bernoulli(means, kl_all)
            ubs_all = self.dup_bernoulli(means, kl_all)

            # base rate P(B)
            (pos_b,), (tot_b,) = self.draw_samples([()], max(100, batch_size))
            base_p = float(pos_b) / float(tot_b)

            rb_all = RuleBatchStats(
                p_a=coverages,
                p_b_given_a=means,
                p_b=base_p,
                lb_prec=lbs_all,
                ub_prec=ubs_all,
                n=n_samples,
            )
            scores_all = objective_fn(rb_all)
            j = int(np.argmax(scores_all))

            best_anchor = anchors[j]
            best_score = float(scores_all[j])
            best_payload = dict(
                score=best_score,
                score_name=objective_label,
                constraint_name=constraint_label,
                p_a=float(coverages[j]),
                p_b=float(base_p),
                p_b_given_a=float(means[j]),
                lb=float(lbs_all[j]),
                ub=float(ubs_all[j]),
                n=int(n_samples[j]),
            )
        else:
            success = True

        meta = self.get_anchor_metadata(best_anchor, success, batch_size=batch_size)

        # single best (backward compatible)
        if best_payload:
            meta["score"] = [best_payload["score"]]
            meta["objective_name"] = best_payload["score_name"]
            meta["constraint_name"] = best_payload["constraint_name"]
            # keep the base rate; only set if not already provided
            if "all_precision" not in meta:
                meta["all_precision"] = base_p
            meta["extra_stats"] = {
                "p_a": best_payload["p_a"],
                "p_b": best_payload["p_b"],
                "p_b_given_a": best_payload["p_b_given_a"],
                "lb_precision": best_payload["lb"],
                "ub_precision": best_payload["ub"],
                "n_samples": best_payload["n"],
            }

        if top_k_return and len(topk_heap) > 0:
            ranked = nlargest(
                min(top_k_return, len(topk_heap)), topk_heap, key=lambda x: x[0]
            )
            meta["candidates"] = []

            # --- NEW: skip the best anchor to avoid duplication
            best_tuple = tuple(best_anchor)

            for _, _, p in ranked:
                a = tuple(p["anchor"])
                if a == best_tuple:
                    continue  # already represented at top level

                # only use get_anchor_metadata if we know it has samples; else manual record
                has_samples = False
                try:
                    # get_anchor_metadata computes precision for each ordered prefix of `a`.
                    # Guard against zero-sample prefixes to avoid division-by-zero in metadata-only path.
                    prefix = tuple()
                    has_samples = True
                    for f in self.state["t_order"].get(a, list(a)):
                        prefix = self._sort(prefix + (f,), allow_duplicates=False)
                        if self.state["t_nsamples"].get(prefix, 0) <= 0:
                            has_samples = False
                            break
                except Exception:
                    has_samples = False

                if has_samples:
                    m = self.get_anchor_metadata(a, True, batch_size=batch_size)
                else:
                    m = {
                        "feature": list(a),
                        "precision": [p["p_b_given_a"]],
                        "coverage": [p["p_a"]],
                        "num_preds": p["n"],
                        "examples": [],
                        "success": True,
                    }

                m["score"] = [p["score"]]
                m["objective_name"] = p["score_name"]
                m["constraint_name"] = p["constraint_name"]
                # don't overwrite if get_anchor_metadata already supplied it
                if "all_precision" not in m:
                    m["all_precision"] = p["p_b"]
                m["extra_stats"] = {
                    "p_a": p["p_a"],
                    "p_b": p["p_b"],
                    "p_b_given_a": p["p_b_given_a"],
                    "lb_precision": p["lb"],
                    "ub_precision": p["ub"],
                    "n_samples": p["n"],
                }
                meta["candidates"].append(m)

            meta["top_k_return"] = len(meta["candidates"])

        self._finalize_instrumentation()
        meta["instrumentation"] = copy.deepcopy(self.instrumentation)
        return meta
