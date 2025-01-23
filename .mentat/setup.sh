curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
. .venv/bin/activate
uv pip install -r requirements.txt