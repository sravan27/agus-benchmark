from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks


def test_attention_distractors_cover_levels():
    tasks = generate_attention_distractor_tasks(AttentionDistractorConfig(count=8, seed=51))
    levels = {task["distractor_level"] for task in tasks}
    assert levels == {0, 1, 2, 3}
    assert tasks[0]["family"] == "attention_distractors"

