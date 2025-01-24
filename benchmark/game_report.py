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
            transition: all 0.3s ease;
        }}
        .submission.winner {{
            background-color: #d4edda;
            border-color: #c3e6cb;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .submission.waiting {{
            background-color: #fff3cd;
            border-color: #ffeeba;
            font-style: italic;
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
    judge_name = players[round.judge].name
    html = f"""
    <div class="round">
        <div class="round-header">
            <h3>Round {round.round_number + 1}</h3>
            <p><strong>Green Card:</strong> "{round.green_card}"</p>
            <p><strong>Judge:</strong> {judge_name}</p>
        </div>
"""

    # Add winner section if round has been decided
    if round.decision:
        winner_name = players[round.decision.winning_player].name
        html += f"""
        <div class="winner-section">
            <h4>🏆 Winner: {winner_name}</h4>
            <p><strong>Winning Card:</strong> "{round.decision.winning_card}"</p>
            <p><strong>Judge's Reasoning:</strong> {round.decision.reasoning}</p>
        </div>"""
    else:
        html += f"""
        <div class="judge-section">
            <p>Waiting for {judge_name} to make a decision...</p>
        </div>"""

    # Add submissions section
    html += """
        <div class="submissions">
            <h4>📝 Submissions:</h4>"""

    # Get all non-judge players
    non_judge_players = [(idx, players[idx]) for idx in players if idx != round.judge]

    # Add each player's submission or waiting message
    for player_idx, player in non_judge_players:
        if player_idx in round.moves:
            move = round.moves[player_idx]
            submission_class = "submission"
            if round.decision and player_idx == round.decision.winning_player:
                submission_class += " winner"
            html += f"""
            <div class="{submission_class}">
                <p><strong>{player.name}'s Card:</strong> "{move.played_card}"</p>
                <div class="thinking">
                    <strong>Reasoning:</strong><br>
                    {move.thinking}
                </div>
            </div>"""
        else:
            html += f"""
            <div class="submission waiting">
                <p>Waiting for {player.name} to play a card...</p>
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
