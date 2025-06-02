import json


def load_models():
    """Load models from models.json file."""
    with open("scripts/models.json", "r") as f:
        data = json.load(f)
        return data.get("models", [])


def create_readme():
    """Create README.md from template and models data."""
    # Load models data
    models = load_models()

    # Read template
    with open("scripts/readme_template.md", "r") as f:
        template = f.read()

    # Generate models table rows
    model_rows = []
    for model in models:
        row = f"| [{model['name']}]({model['url']}) | `{model['size']}` | `{model['task']}` | `{model['category']}` | {model['description']} |"
        model_rows.append(row)

    # Join rows with newlines
    models_table = "\n".join(model_rows)

    # Replace placeholder in template
    readme_content: str = template.replace("{{MODELS_TABLE}}", models_table)
    print(readme_content)
    # Write to README.md
    with open("readme.md", "w") as f:
        f.write(readme_content)


if __name__ == "__main__":
    create_readme()
