from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks


def test_social_miniworlds_capture_false_belief():
    tasks = generate_social_miniworld_tasks(SocialMiniworldConfig(count=5, seed=67))
    first = tasks[0]
    assert first["family"] == "social_miniworlds"
    assert "actual_location" in first["answer"]
    assert "belief_of_false_belief_agent" in first["answer"]
    assert "most_reliable_agent" in first["answer"]
