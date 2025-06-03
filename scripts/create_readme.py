import json
from pathlib import Path


def load_models():
    """Load models from models.json file."""
    models_path = Path(__file__).parent / "models.json"
    with open(models_path, "r") as f:
        data = json.load(f)
        return data.get("models", [])


def load_categories():
    """Load categories from categories.json file."""
    categories_path = Path(__file__).parent / "categories.json"
    with open(categories_path, "r") as f:
        data = json.load(f)
        return data.get("categories", [])


def create_readme():
    """Create README.md from template and models data."""
    # Load models data
    models = load_models()
    categories = load_categories()

    # Read template
    template_path = Path(__file__).parent / "readme_template.md"
    with open(template_path, "r") as f:
        template = f.read()

    # Sort models based on category and task
    models.sort(key=lambda x: (x["category"], x["task"], x["size"]))

    # Group models by task
    models_by_task = {}
    for model in models:
        task = model["task"]
        if task not in models_by_task:
            models_by_task[task] = []
        models_by_task[task].append(model)

    # Generate tables for each task
    task_tables = []
    for task, task_models in models_by_task.items():
        # Create header for this task
        task_header = f"\n### {task.title()} Models\n"

        # Generate table rows for this task
        model_rows = [
            "| Name | Size | Category | Sub Category |",
            "|------|------|----------|--------------|",
        ]
        for model in task_models:
            row = f"| [{model['name']}]({model['url']}) | `{model['size']}` | `{", ".join(model['category'])}` | `{", ".join(model['sub_category'])}` |"
            model_rows.append(row)

        # Combine header and table for this task
        task_table = task_header + "\n".join(model_rows)
        task_tables.append(task_table)

    # Join all task tables with newlines
    models_table = "\n".join(task_tables)

    # Generate categories table rows
    category_rows = [
        "| Name | Description |",
        "|------|-------------|",
    ]

    # Dictionary to store sub-category tables by category
    sub_category_tables = {}

    for category in categories:
        sub_categories = []
        # Create sub-category table for this category
        sub_category_rows = [
            f"\n### {category['name']}\n",
            "| Sub Category | Description |",
            "|--------------|-------------|",
        ]

        # Add sub-category rows
        for sub_cat in category["sub_categories"]:
            sub_cat_row = f"| `{sub_cat['name']}` | {sub_cat['description']} |"
            sub_category_rows.append(sub_cat_row)
            sub_categories.append(sub_cat["name"])

        # Get models for this category
        category_models = []
        for model in models:
            if category["name"] in model["category"]:
                model_row = f"| [{model['name']}]({model['url']}) | `{model['size']}` | `{model['task']}` |"
                category_models.append(model_row)

        # Create dropdown for models
        if category_models:
            models_dropdown = [
                "\n<details>",
                f"<summary>Models in {category['name']}</summary>\n",
                "| Name | Size | Task |",
                "|------|------|------|",
                *category_models,
                "</details>\n",
            ]
            sub_category_rows.extend(models_dropdown)

        # Store the sub-category table for this category
        sub_category_tables[category["name"]] = "\n".join(sub_category_rows)

        # Add main category row
        category_row = f"| `{category['name']}` | {category['description']} |"
        category_rows.append(category_row)

    # Create main categories table
    categories_table = "\n".join(category_rows)

    # Combine all sub-category tables
    sub_categories_table = "\n".join(sub_category_tables.values())

    # Replace both placeholders in template
    readme_content = template.replace("{{CATEGORIES_TABLE}}", categories_table)
    readme_content = readme_content.replace(
        "{{SUB_CATEGORIES_TABLE}}", sub_categories_table
    )

    # Replace placeholder in template
    readme_content: str = readme_content.replace("{{MODELS_TABLE}}", models_table)

    readme_path = Path(__file__).parent.parent / "readme.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)


if __name__ == "__main__":
    create_readme()
