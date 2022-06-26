SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: help
.DEFAULT_GOAL=help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: jupyter-lab
jupyter:  ## start jupyter lab
	poetry run jupyter lab --ip=127.0.0.1 --no-browser --notebook-dir=notebooks

.PHONY: pytest
pytest:  ## run pytest
	poetry run python -m pytest src

.PHONY: mypy
mypy:  ## run mypy
	poetry run python -m mypy src


