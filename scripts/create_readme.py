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

    # Generate models table rows
    model_rows: list[str] = [
        "| Name | Size | Task | Category | Sub Category | Description |",
        "|------|------|------|----------|--------------|-------------|",
    ]
    for model in models:
        row = f"| [{model['name']}]({model['url']}) | `{model['size']}` | `{model['task']}` | `{model['category']}` | `{model['sub_category']}` | {model['description']} |"
        model_rows.append(row)

    # Join rows with newlines
    models_table = "\n".join(model_rows)

    # Generate categories table rows
    category_rows = [
        "| Name | Description | Sub Categories |",
        "|------|-------------|----------------|",
    ]
    sub_category_rows = [
        "| Category | Sub Category | Description |",
        "|--------------|--------------|-------------|",
    ]
    for category in categories:
        sub_categories = []
        # Add sub-category rows
        for sub_cat in category["sub_categories"]:
            sub_cat_row = f"| `{category['name']} | {sub_cat['name']}` | {sub_cat['description']} |"
            sub_category_rows.append(sub_cat_row)
            sub_categories.append(sub_cat["name"])

        # Add main category row
        category_row = f"| `{category['name']}` | {category['description']} | {', '.join(sub_categories)} |"
        category_rows.append(category_row)

    # Create separate tables
    categories_table = "\n".join(category_rows)
    sub_categories_table = "\n".join(sub_category_rows)

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
