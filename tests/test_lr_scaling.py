"""Regression tests for persisted batch-size learning-rate corrections."""

from nanochat.common import add_effective_learning_rates, inherited_learning_rate


def test_effective_rates_match_optimizer_inputs_without_mutating_cli_config():
    user_config = {
        "embedding_lr": 0.3,
        "unembedding_lr": 0.008,
        "matrix_lr": 0.02,
        "scalar_lr": 0.5,
    }

    persisted = add_effective_learning_rates(user_config, 2.0)

    assert user_config == {
        "embedding_lr": 0.3,
        "unembedding_lr": 0.008,
        "matrix_lr": 0.02,
        "scalar_lr": 0.5,
    }
    assert persisted["batch_lr_scale"] == 2.0
    assert persisted["effective_embedding_lr"] == 0.6
    assert persisted["effective_unembedding_lr"] == 0.016
    assert persisted["effective_matrix_lr"] == 0.04
    assert persisted["effective_scalar_lr"] == 1.0


def test_inheritance_prefers_effective_rate_and_supports_legacy_checkpoints():
    assert inherited_learning_rate(
        {"embedding_lr": 0.3, "effective_embedding_lr": 0.45},
        "embedding_lr",
        0.1,
    ) == 0.45
    assert inherited_learning_rate({"embedding_lr": 0.3}, "embedding_lr", 0.1) == 0.3
    assert inherited_learning_rate({}, "embedding_lr", 0.1) == 0.1
