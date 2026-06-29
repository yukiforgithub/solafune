.PHONY: dev-env

dev-env:
	@echo "Upgrading pip..."
	@python -m pip install --user --upgrade pip
	@echo "Installing uv..."
	@python -m pip install --user --upgrade uv
	@echo "Generating virtual environment..."
	@uv venv -p python3.13 --allow-existing
	@echo "Installing dev environment in editable mode..."
	@uv pip install -e .[dev]
	@echo "Activate dev environment with:"
	@echo "source .venv/bin/activate"