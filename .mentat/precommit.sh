. .venv/bin/activate
pip install -r requirements.txt
pip install aiohttp[speedups]
ruff format .
ruff check --fix .
pyright