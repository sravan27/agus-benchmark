from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks


def test_hidden_rule_generation_shape():
    tasks = generate_hidden_rule_tasks(HiddenRuleConfig(count=5, seed=11))
    assert len(tasks) == 5
    first = tasks[0]
    assert first["family"] == "hidden_rule"
    assert len(first["query"]["induction_queries"]) == 2
    assert len(first["query"]["shift_queries"]) == 4
    assert len(first["answer"]["shift_targets"]) == 4

