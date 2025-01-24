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
            padding: 10px;
            line-height: 1.4;
        }}
        .header {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .round {{
            border: 1px solid #dee2e6;
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .round-header {{
            background-color: #e9ecef;
            padding: 8px 10px;
            margin: -10px -10px 10px -10px;
            border-radius: 8px 8px 0 0;
            border-bottom: 1px solid #dee2e6;
        }}
        .winner-section {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 8px 10px;
            margin-bottom: 10px;
            border-radius: 6px;
        }}
        .judge-section {{
            background-color: #e2e3e5;
            border: 1px solid #d6d8db;
            padding: 8px 10px;
            margin-bottom: 10px;
            border-radius: 6px;
        }}
        .submissions {{
            margin-top: 10px;
        }}
        .submission {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 8px 10px;
            margin-bottom: 8px;
            border-radius: 6px;
        }}
        .thinking {{
            margin-top: 6px;
            padding: 6px 8px;
            background-color: #fff;
            border-left: 3px solid #6c757d;
            font-style: italic;
        }}
    </style>
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
        <div class="round-header">
            <h3>Round {round.round_number + 1}</h3>
            <p><strong>Green Card:</strong> {round.green_card}</p>
        </div>
"""

    # Add winner section if round has been decided
    if round.decision:
        html += f"""
        <div class="winner-section">
            <h4>🏆 Winner: {players[round.decision.winning_player].name}</h4>
            <p><strong>Winning Card:</strong> {round.decision.winning_card}</p>
        </div>"""

    # Add judge section
    html += f"""
        <div class="judge-section">
            <h4>👨‍⚖️ Judge: {players[round.judge].name}</h4>"""
    if round.decision:
        html += f"""
            <p><strong>Reasoning:</strong> {round.decision.reasoning}</p>"""
    html += """
        </div>"""

    # Add submissions section
    html += """
        <div class="submissions">
            <h4>📝 Submissions:</h4>"""

    # Add each player's submission (excluding judge)
    for player_idx, move in round.moves.items():
        if player_idx != round.judge:  # Only show non-judge players' submissions
            html += f"""
            <div class="submission">
                <p><strong>Player:</strong> {players[player_idx].name}</p>
                <p><strong>Card Played:</strong> {move.played_card}</p>
                <div class="thinking">
                    <strong>Reasoning:</strong><br>
                    {move.thinking}
                </div>
            </div>"""

    html += """
        </div>
    </div>"""
    return html


def save_html_report(game: Game, filepath: str) -> None:
    """Save the game report as an HTML file"""
    report = generate_html_report(game)
    with open(filepath, "w") as f:
        f.write(report)
