from typing import Dict
from benchmark.game import Game, Round


def generate_html_report(game: Game) -> str:
    """Generate an HTML report for the game"""

    # Calculate stats for header
    total_rounds = len(game.rounds)
    player_stats = {
        idx: {"name": player.name, "wins": len(player.won_rounds)}
        for idx, player in game.players.items()
    }
    standings = sorted(player_stats.items(), key=lambda x: x[1]["wins"], reverse=True)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Game Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .round {{
            border: 1px solid #ddd;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
        }}
        .move {{
            margin: 10px 0;
            padding: 10px;
            background-color: #f9f9f9;
        }}
        .thinking {{
            display: none;
            margin-top: 10px;
            padding: 10px;
            background-color: #fff;
            border-left: 3px solid #ccc;
        }}
        .show-thinking {{
            color: blue;
            text-decoration: underline;
            cursor: pointer;
        }}
        .winner {{
            background-color: #e6ffe6;
        }}
    </style>
    <script>
        function toggleThinking(moveId) {{
            const thinking = document.getElementById(moveId);
            if (thinking.style.display === 'none') {{
                thinking.style.display = 'block';
            }} else {{
                thinking.style.display = 'none';
            }}
        }}
    </script>
</head>
<body>
    <div class="header">
        <h1>Game Report</h1>
        <h2>Stats</h2>
        <p>Total Rounds: {total_rounds}</p>
        <h3>Standings:</h3>
        <ul>
"""

    # Add standings to header
    for idx, stats in standings:
        html += f"""
            <li>{stats['name']}: {stats['wins']} win{'s' if stats['wins'] != 1 else ''}</li>"""

    html += """
        </ul>
    </div>
    <h2>Rounds</h2>
"""

    # Add each round
    for round in game.rounds:
        html += _generate_round_html(round, game.players, player_stats)

    html += """
</body>
</html>
"""
    return html


def _generate_round_html(round: Round, players: Dict, player_stats: Dict) -> str:
    """Generate HTML for a single round"""
    html = f"""
    <div class="round">
        <h3>Round {round.round_number + 1}</h3>
        <p><strong>Green Card:</strong> {round.green_card}</p>
        <p><strong>Judge:</strong> {players[round.judge].name}</p>
        <h4>Moves:</h4>
"""

    # Add each move
    for player_idx, move in round.moves.items():
        is_winner = round.decision and round.decision.winning_player == player_idx
        winner_class = "winner" if is_winner else ""

        html += f"""
        <div class="move {winner_class}">
            <p><strong>{players[player_idx].name}</strong></p>
            <p>Played: {move.played_card}</p>
            <span class="show-thinking" onclick="toggleThinking('thinking-{round.round_number}-{player_idx}')">
                Show/Hide Thinking
            </span>
            <div id="thinking-{round.round_number}-{player_idx}" class="thinking">
                {move.thinking}
            </div>
        </div>"""

    # Add round decision if it exists
    if round.decision:
        html += f"""
        <div class="move winner">
            <h4>Winner: {players[round.decision.winning_player].name}</h4>
            <p>Winning Card: {round.decision.winning_card}</p>
            <p>Judge's Reasoning: {round.decision.reasoning}</p>
        </div>"""

    html += """
    </div>"""
    return html


def save_html_report(game: Game, filepath: str) -> None:
    """Save the game report as an HTML file"""
    report = generate_html_report(game)
    with open(filepath, "w") as f:
        f.write(report)
