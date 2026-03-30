from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks


def test_shift_transfer_has_distinct_representations():
    tasks = generate_shift_transfer_tasks(ShiftTransferConfig(count=3, seed=23))
    first = tasks[0]
    assert first["family"] == "shift_transfer"
    assert first["query"]["source_query"]["input"] != first["query"]["transfer_query"]["input"]
    assert len(first["answer"]["source_target"]) == len(first["answer"]["transfer_target"])

