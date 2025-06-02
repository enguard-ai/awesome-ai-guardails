import json


def _open_models_json() -> dict:
    """Open the models.json file from the awesome-ai-guardrails repository."""
    models_path = "./models.json"
    with open(models_path, "r") as f:
        return json.load(f)


AVAILABLE_MODELS = _open_models_json()["models"]


def _get_task_types(models: dict) -> list[str]:
    """Get the task types for a given model."""
    return set([model["task"] for model in models])


AVAILABLE_TASK_TYPES = _get_task_types(AVAILABLE_MODELS)


def _get_guardrail_categories(models: dict) -> tuple[set[str], set[str]]:
    """Get the guardrail categories for a given model."""
    return set([model["category"] for model in models]), set(
        [model["sub_category"] for model in models]
    )


AVAILABLE_GUARDRAIL_CATEGORIES, AVAILABLE_GUARDRAIL_SUB_CATEGORIES = (
    _get_guardrail_categories(AVAILABLE_MODELS)
)


def get_guardrails_from_task(
    task: str,
) -> list[dict]:
    """Get the guardrail from a task."""
    if task not in AVAILABLE_TASK_TYPES:
        raise ValueError(f"Task {task} not found")
    potential_guardrails = []
    for model in AVAILABLE_MODELS:
        if model["task"] == task:
            potential_guardrails.append(model["task"])
    return potential_guardrails


def get_guardrail_from_category(
    category: str | None = None,
    sub_category: str | None = None,
) -> list[dict]:
    """Get the guardrail from a category or sub-category."""
    if category is not None and category not in AVAILABLE_GUARDRAIL_CATEGORIES:
        raise ValueError(f"Category {category} not found")
    if (
        sub_category is not None
        and sub_category not in AVAILABLE_GUARDRAIL_SUB_CATEGORIES
    ):
        raise ValueError(f"Sub-category {sub_category} not found")
    for model in AVAILABLE_MODELS:
        if (category is None or category in model["category"]) and (
            sub_category is None or sub_category in model["sub_category"]
        ):
            return model
