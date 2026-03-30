"""Social cognition miniworld task family."""

from __future__ import annotations

from dataclasses import dataclass

from src.schemas.task_schema import AGUSTask
from src.utils.seeds import make_rng

AGENT_POOL = (
    "Ava",
    "Ben",
    "Cyra",
    "Dax",
    "Eli",
    "Faye",
    "Gio",
    "Hana",
)
OBJECTS = ("token", "journal", "key", "badge", "map", "coin")
LOCATION_SETS = (
    ("red locker", "blue locker", "green locker"),
    ("north room", "east room", "west room"),
    ("glass box", "wooden box", "metal box"),
    ("table drawer", "desk drawer", "cabinet drawer"),
)


@dataclass(frozen=True)
class SocialMiniworldConfig:
    """Generation settings for social cognition tasks."""

    count: int = 100
    seed: int = 67


def generate_social_miniworld_tasks(cfg: SocialMiniworldConfig) -> list[dict]:
    """Generate miniworlds with partial knowledge, false belief, and incentives."""
    rng = make_rng(cfg.seed)
    tasks: list[dict] = []

    for idx in range(cfg.count):
        num_agents = 4 if idx % 3 == 0 else 3
        agents = rng.sample(AGENT_POOL, k=num_agents)
        object_name = rng.choice(OBJECTS)
        locations = list(LOCATION_SETS[idx % len(LOCATION_SETS)])

        initial_location, actual_location = rng.sample(locations, 2)
        witness = rng.choice(agents)
        false_belief_agent = rng.choice([agent for agent in agents if agent != witness])
        remaining = [agent for agent in agents if agent not in {witness, false_belief_agent}]
        deceiver = rng.choice(remaining) if remaining else false_belief_agent
        rumor_agent = remaining[0] if len(remaining) > 1 else None
        decoy_location = rng.choice([loc for loc in locations if loc not in {actual_location, initial_location}])

        examples = [
            {
                "phase": "shared_observation",
                "event": f"Everyone first saw the {object_name} placed in the {initial_location}.",
            },
            {
                "phase": "hidden_move",
                "event": f"Later, the {object_name} was moved to the {actual_location}. Only {witness} saw the move.",
            },
            {
                "phase": "statement",
                "agent": witness,
                "content": f'The {object_name} is now in the {actual_location}.',
                "knowledge_access": "direct_observation",
                "incentive": "help_the_search",
            },
            {
                "phase": "statement",
                "agent": deceiver,
                "content": f'The {object_name} is in the {decoy_location}.',
                "knowledge_access": "no_direct_observation",
                "incentive": "wins_if_others_search_the_wrong_place",
            },
            {
                "phase": "belief_state",
                "agent": false_belief_agent,
                "content": (
                    f"{false_belief_agent} did not see the move and has only the original placement "
                    "unless a trustworthy witness updates them."
                ),
            },
        ]
        if rumor_agent is not None:
            examples.append(
                {
                    "phase": "statement",
                    "agent": rumor_agent,
                    "content": f'I did not see the move, but {witness} says the {object_name} is in the {actual_location}.',
                    "knowledge_access": "second_hand_report",
                    "incentive": "neutral",
                }
            )

        task = AGUSTask(
            task_id=f"social_miniworlds_{idx:04d}",
            family="social_miniworlds",
            difficulty="hard" if rumor_agent is not None else "medium",
            context={
                "instruction": (
                    "Reason about who knows what, who has stale beliefs, and whose statement should be trusted."
                ),
                "setting": f"A small group is trying to locate a {object_name}.",
                "agents": agents,
            },
            examples=examples,
            query={
                "questions": [
                    {"id": "actual_location", "prompt": f"Where is the {object_name} actually located now?"},
                    {
                        "id": "false_belief",
                        "prompt": f"Where does {false_belief_agent} still believe the {object_name} is?",
                    },
                    {
                        "id": "best_informant",
                        "prompt": "Which agent should a newcomer trust most for the current location?",
                    },
                ]
            },
            answer={
                "actual_location": actual_location,
                "belief_of_false_belief_agent": initial_location,
                "most_reliable_agent": witness,
            },
            metadata={
                "witness": witness,
                "false_belief_agent": false_belief_agent,
                "deceiver": deceiver,
                "rumor_agent": rumor_agent,
                "object_name": object_name,
                "locations": locations,
                "theory_of_mind_order": 1,
            },
            latent_rule_summary="Agents differ in observation access, incentives, and beliefs after a hidden move.",
            shift_type="belief_asymmetry",
            distractor_level=0,
            scoring_notes=[
                "Score actual location, false-belief inference, and trust selection separately.",
                "These tasks target social reasoning rather than lexical overlap.",
            ],
        )
        tasks.append(task.to_dict())

    return tasks
