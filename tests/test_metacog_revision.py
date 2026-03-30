from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks


def test_metacog_revision_has_revision_target():
    tasks = generate_metacog_revision_tasks(MetacogRevisionConfig(count=4, seed=37))
    first = tasks[0]
    assert first["family"] == "metacog_revision"
    assert first["answer"]["should_revise"] is True
    assert len(first["answer"]["acceptable_initial_targets"]) >= 1
    assert "expected_initial_certainty" in first["metadata"]

