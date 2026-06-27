# saccade-world-model-exp — 開発タスク (uv + just)。
# GitHub CI は持たず、`just run` をコミット前のローカル品質ゲートとして使う。

# Default recipe (shows help)
default:
    @just --list

# Install dependencies and the pre-commit git hook.
setup:
    uv sync
    uv run pre-commit install

# Run all pre-commit hooks (ruff check + format, file hygiene, ...).
format:
    uv run pre-commit run -a

# Lint only (no autofix).
lint:
    uv run ruff check

# Static type check.
type:
    uv run pyright

# Run the test suite.
test:
    uv run pytest -v

# Local quality gate: format -> test -> type. Run before committing.
run: format test type

# Remove caches and build artifacts.
clean:
    find . -type d -name "__pycache__" -prune -exec rm -rf {} +
    rm -rf .pytest_cache .ruff_cache
