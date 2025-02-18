. .venv/bin/activate
pip install -r requirements.txt
ruff format .
ruff check --fix .
pyright