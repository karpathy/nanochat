"""Checkpoint hyperparameter inheritance, without importing the training runtime."""

import copy
from argparse import Namespace
import unittest

from scripts.sft_config import inherit_sft_config


class TestSFTConfig(unittest.TestCase):
    def resolve(self, meta, **overrides):
        args = Namespace(max_seq_len=None, device_batch_size=None, total_batch_size=None,
                         embedding_lr=None, unembedding_lr=None, matrix_lr=None)
        vars(args).update(overrides)
        inherit_sft_config(args, meta, lambda message: None)
        return args

    def test_inherits_effective_rates_at_checkpoint_batch_size(self):
        # Hand-derived from pretraining's sqrt(batch / 524288) scaling.
        for batch, expected in [
            (131072, (0.15, 0.002, 0.01)),
            (524288, (0.3, 0.004, 0.02)),
            (2097152, (0.6, 0.008, 0.04)),
        ]:
            with self.subTest(batch=batch):
                meta = {"total_batch_size": batch, "user_config": {
                    "total_batch_size": -1,
                    "embedding_lr": 0.3, "unembedding_lr": 0.004, "matrix_lr": 0.02,
                }}
                before = copy.deepcopy(meta)
                args = self.resolve(meta)
                self.assertEqual((args.embedding_lr, args.unembedding_lr, args.matrix_lr), expected)
                self.assertEqual(meta, before)

    def test_sft_batch_override_does_not_rescale_checkpoint_rates(self):
        args = self.resolve({"total_batch_size": 2097152, "user_config": {"matrix_lr": 0.02}},
                            total_batch_size=131072)
        self.assertEqual(args.total_batch_size, 131072)
        self.assertEqual(args.matrix_lr, 0.04)
        self.assertEqual((args.embedding_lr, args.unembedding_lr), (0.3, 0.004))

    def test_each_explicit_lr_override_is_literal(self):
        rates = {"embedding_lr": 0.3, "unembedding_lr": 0.004, "matrix_lr": 0.02}
        for name in rates:
            for override in (0.0, 0.123):
                with self.subTest(name=name, override=override):
                    args = self.resolve({"total_batch_size": 2097152, "user_config": rates},
                                        **{name: override})
                    self.assertEqual(getattr(args, name), override)
                    other = "matrix_lr" if name != "matrix_lr" else "embedding_lr"
                    self.assertEqual(getattr(args, other), {"matrix_lr": 0.04, "embedding_lr": 0.6}[other])

    def test_missing_historical_metadata_keeps_defaults(self):
        for meta in ({}, {"total_batch_size": 2097152},
                     {"total_batch_size": 2097152, "user_config": {
                         "embedding_lr": None, "unembedding_lr": None, "matrix_lr": None}}):
            with self.subTest(meta=meta):
                args = self.resolve(meta)
                self.assertEqual((args.embedding_lr, args.unembedding_lr, args.matrix_lr),
                                 (0.3, 0.004, 0.02))
                self.assertEqual((args.max_seq_len, args.device_batch_size), (2048, 32))

    def test_raw_rates_without_resolved_batch_are_not_scaled(self):
        for batch_meta in ({}, {"total_batch_size": None}):
            with self.subTest(batch_meta=batch_meta):
                args = self.resolve({**batch_meta, "user_config": {
                    "total_batch_size": -1, "matrix_lr": 0.06}})
                self.assertEqual(args.matrix_lr, 0.06)
                self.assertEqual(args.total_batch_size, 524288)

    def test_reusing_metadata_does_not_scale_rates_twice(self):
        meta = {"total_batch_size": 2097152, "user_config": {"matrix_lr": 0.02}}
        self.assertEqual(self.resolve(meta).matrix_lr, 0.04)
        self.assertEqual(self.resolve(meta).matrix_lr, 0.04)

    def test_present_zero_rate_is_not_replaced_by_default(self):
        args = self.resolve({"total_batch_size": 2097152, "user_config": {"matrix_lr": 0.0}})
        self.assertEqual(args.matrix_lr, 0.0)


if __name__ == "__main__":
    unittest.main()
