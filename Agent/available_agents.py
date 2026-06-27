"""Available battle agent options exposed to web frontends.

The names in these lists are the exact model_type tokens accepted by
`AgentManager.load_agent` in realtime controllers.
"""

CHESS_AGENT_NAMES = [
    "Random_Agent",
    "Human_Agent",
    "Rule_Agent",
    "Think_Agent",
    "FastOnly_Agent",
    "SlowOnly_Agent",
    "Think_WithoutOpponent_Agent",
    "Think_WithoutSelf_Agent",
    "LLM_Agent",
    "CoT_Agent",
    "Reflexion_Agent",
    "MemoryLLM_Agent",
    "CodingPairs_Agent",
]


MAZE_AGENT_NAMES = [
    "Random_Agent",
    "Human_Agent",
    "Rule_Agent",
    "Think_Agent",
    "FastOnly_Agent",
    "SlowOnly_Agent",
    "Think_WithoutOpponent_Agent",
    "Think_WithoutSelf_Agent",
    "LLM_Agent",
    "CoT_Agent",
    "Reflexion_Agent",
    "MemoryLLM_Agent",
    "CodingPairs_Agent",
]


def build_agent_groups(agent_names: list[str], llm_models: list[str], default_llm: str | None = None) -> list[dict]:
    names = set(agent_names or [])

    def _opts(items: list[tuple[str, str]]) -> list[dict]:
        return [{"value": v, "label": l} for v, l in items if v in names]

    groups: list[dict] = []

    groups.append({
        "key": "human",
        "label": "Human_Agent",
        "options": _opts([
            ("Human_Agent", "Human_Agent"),
        ]),
    })

    groups.append({
        "key": "rule",
        "label": "Rule_Agent",
        "options": _opts([
            ("Random_Agent", "Random_Agent"),
            ("Rule_Agent", "Rule_Agent"),
        ]),
    })

    llm_base_options = [
        {
            "value": f"LLM_Agent:{m}",
            "label": m,
            "model": m,
            "agent": "LLM_Agent",
            "is_llm_model": True,
            "is_default": (m == default_llm),
        }
        for m in (llm_models or [])
    ]
    if "LLM_Agent" not in names:
        llm_base_options = []

    groups.append({
        "key": "llm_base",
        "label": "LLM_base_Agent",
        "options": llm_base_options,
    })

    groups.append({
        "key": "llm_farm",
        "label": "LLM_farm_Agent",
        "options": _opts([
            ("CoT_Agent", "CoT_Agent"),
            ("Reflexion_Agent", "Reflexion_Agent"),
            ("MemoryLLM_Agent", "MemoryLLM_Agent"),
            ("CodingPairs_Agent", "CodingPairs_Agent"),
        ]),
    })

    groups.append({
        "key": "r_thinker",
        "label": "R_Thinker",
        "options": _opts([
            ("Think_Agent", "Think_Agent"),
        ]),
    })

    groups.append({
        "key": "r_thinker_variants",
        "label": "R_Thinker_variants",
        "options": _opts([
            ("FastOnly_Agent", "FastOnly_Agent"),
            ("SlowOnly_Agent", "SlowOnly_Agent"),
            ("Think_WithoutOpponent_Agent", "Think_WithoutOpponent_Agent"),
            ("Think_WithoutSelf_Agent", "Think_WithoutSelf_Agent"),
        ]),
    })

    return [g for g in groups if g.get("options")]
