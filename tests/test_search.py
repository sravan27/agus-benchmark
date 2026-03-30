from src.search.probe_conditioned_search import (
    build_search_space,
    evaluate_search_space,
    rank_family_results,
    run_probe_conditioned_search,
)


def test_search_space_is_deterministic():
    left = build_search_space(count_per_config=4)
    right = build_search_space(count_per_config=4)

    assert left == right
    assert set(left) == {"attention_distractors", "shift_transfer"}
    assert len(left["attention_distractors"]) >= 4
    assert len(left["shift_transfer"]) >= 4


def test_ranking_logic_prefers_higher_weighted_score_then_signal():
    ranked = rank_family_results(
        [
            {
                "config_id": "cfg_low",
                "search_metrics": {
                    "curation_weighted_score": 0.61,
                    "average_benchmark_signal": 0.8,
                    "kept_rate": 0.7,
                    "reject_rate": 0.1,
                },
            },
            {
                "config_id": "cfg_high",
                "search_metrics": {
                    "curation_weighted_score": 0.72,
                    "average_benchmark_signal": 0.75,
                    "kept_rate": 0.6,
                    "reject_rate": 0.05,
                },
            },
            {
                "config_id": "cfg_tie_break",
                "search_metrics": {
                    "curation_weighted_score": 0.72,
                    "average_benchmark_signal": 0.78,
                    "kept_rate": 0.58,
                    "reject_rate": 0.05,
                },
            },
        ]
    )

    assert ranked[0]["config_id"] == "cfg_tie_break"
    assert ranked[1]["config_id"] == "cfg_high"


def test_attention_family_search_returns_ranked_configs():
    candidates = build_search_space(families=("attention_distractors",), count_per_config=4)["attention_distractors"]
    ranked = evaluate_search_space("attention_distractors", candidates)

    assert ranked
    assert ranked[0]["family"] == "attention_distractors"
    assert ranked[0]["search_metrics"]["family_specific_metric"] == "distractor_discrimination_score"
    assert "adversarial_query_mode" in ranked[0]["config"]


def test_shift_family_search_returns_ranked_configs():
    candidates = build_search_space(families=("shift_transfer",), count_per_config=4)["shift_transfer"]
    ranked = evaluate_search_space("shift_transfer", candidates)

    assert ranked
    assert ranked[0]["family"] == "shift_transfer"
    assert ranked[0]["search_metrics"]["family_specific_metric"] == "transfer_depth_score"
    assert "bridge_representation_mode" in ranked[0]["config"]


def test_search_run_writes_best_config_artifacts(tmp_path):
    results = run_probe_conditioned_search(
        count_per_config=4,
        emit_best_batches=True,
        best_batch_size=3,
        output_dir=tmp_path,
    )

    assert (tmp_path / "search_results.json").exists()
    assert (tmp_path / "best_generator_configs.json").exists()
    assert (tmp_path / "search_summary.json").exists()
    assert (tmp_path / "best_batches" / "attention_distractors.json").exists()
    assert (tmp_path / "best_batches" / "shift_transfer.json").exists()
    assert results["search_summary"]["families"]["attention_distractors"]["winner"]
    assert results["search_summary"]["families"]["shift_transfer"]["fallback"]
