# Auto_ML

python3 -m venv vautoml

source vautoml/bin/activate

docker compose up -d --force-recreate 

python -m runner.cli ingest-step