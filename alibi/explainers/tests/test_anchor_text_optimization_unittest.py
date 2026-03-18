import unittest
import numpy as np
import spacy

from alibi.explainers.anchors.anchor_text import AnchorText


class TestAnchorTextOptimization(unittest.TestCase):

    @staticmethod
    def _predictor(texts):
        out = []
        for t in texts:
            lt = t.lower()
            out.append(1 if ("good" in lt and "not" not in lt) else 0)
        return np.asarray(out)

    def setUp(self):
        self.nlp = spacy.blank("en")
        self.text = "this is a good movie with clear plot"

    def _build(self, seed=0):
        return AnchorText(
            predictor=self._predictor,
            sampling_strategy='unknown',
            nlp=self.nlp,
            seed=seed,
            sample_proba=0.5,
        )

    def test_instrumentation_exists_for_anchor_text(self):
        explainer = self._build(seed=1)
        exp = explainer.explain(
            self.text,
            threshold=0.8,
            batch_size=32,
            coverage_samples=200,
            max_perturbation_batch_size=16,
            max_total_samples=512,
        )
        self.assertIn('instrumentation', exp.raw)
        inst = exp.raw['instrumentation']
        self.assertGreater(inst['time_per_explanation_s'], 0)
        self.assertGreater(inst['model_calls'], 0)
        self.assertGreater(inst['perturbation_samples_evaluated'], 0)
        self.assertIn('predictor_invocations', inst)
        self.assertIn('predicted_samples_total', inst)
        self.assertGreater(inst['predictor_invocations'], 0)
        self.assertGreater(inst['predicted_samples_total'], 0)

    def test_deterministic_seed_anchor_text(self):
        explainer_a = self._build(seed=7)
        explainer_b = self._build(seed=7)
        kwargs = dict(threshold=0.8, batch_size=32, coverage_samples=200, min_samples_start=32)
        exp_a = explainer_a.explain(self.text, **kwargs)
        exp_b = explainer_b.explain(self.text, **kwargs)
        self.assertEqual(exp_a.anchor, exp_b.anchor)
        self.assertTrue(np.isclose(exp_a.precision, exp_b.precision))

    def test_high_budget_baseline_vs_optimized_match(self):
        explainer_base = self._build(seed=11)
        explainer_opt = self._build(seed=11)
        base_kwargs = dict(threshold=0.8, batch_size=32, coverage_samples=200, min_samples_start=64,
                           max_total_samples=100000, stream_text_sampling=False)
        opt_kwargs = dict(threshold=0.8, batch_size=32, coverage_samples=200, min_samples_start=64,
                          max_total_samples=100000, max_perturbation_batch_size=1024,
                          memory_saver_mode=False, adaptive_budget=False, stream_text_sampling=False)
        exp_base = explainer_base.explain(self.text, **base_kwargs)
        exp_opt = explainer_opt.explain(self.text, **opt_kwargs)
        self.assertEqual(exp_base.anchor, exp_opt.anchor)
        self.assertTrue(np.isclose(exp_base.precision, exp_opt.precision))

    def test_memory_saver_mode_anchor_text(self):
        explainer = self._build(seed=3)
        exp = explainer.explain(
            self.text,
            threshold=0.8,
            batch_size=32,
            coverage_samples=200,
            memory_saver_mode=True,
            max_total_samples=300,
        )
        self.assertIn('instrumentation', exp.raw)
        self.assertGreaterEqual(exp.coverage, 0)


if __name__ == '__main__':
    unittest.main()
