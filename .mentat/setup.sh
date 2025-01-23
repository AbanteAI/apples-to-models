curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
$HOME/.local/bin/uv venv
. .venv/bin/activate
uv pip install -r requirements.txt