PROJECT=frailty-automl-poc

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f

bash-runner:
	docker compose exec runner bash

bash-orch:
	docker compose exec orchestrator bash

lint:
	docker compose exec runner ruff runner orchestrator

pipeline:
	docker compose exec runner bash python orchestrator/main.py
