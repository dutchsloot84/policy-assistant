.PHONY: run-api run-ui test lint fmt

run-api:
	uvicorn src.api.app:app --reload --port 8000

run-ui:
	streamlit run src/ui/app_streamlit.py

test:
	pytest -q --cov=src

lint:
	ruff check . && black --check . && mypy src

fmt:
	black .
