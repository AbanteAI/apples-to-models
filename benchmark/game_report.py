import os
import tempfile
from typing import Dict

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from benchmark.game import Game, Round


def generate_cumulative_wins_chart(game: Game) -> str:
    """Generate a line chart showing cumulative wins over rounds for each player"""
    plt.figure(figsize=(10, 6))

    # Calculate cumulative wins for each player
    player_wins = {idx: [0] for idx in game.players.keys()}
    for round_num, round in enumerate(game.rounds):
        if round.decision:
            winner = round.decision.winning_player
            for player_id in game.players.keys():
                player_wins[player_id].append(
                    player_wins[player_id][-1] + (1 if player_id == winner else 0)
                )
        else:
            # If round not decided, repeat last value
            for player_id in game.players.keys():
                player_wins[player_id].append(player_wins[player_id][-1])

    # Plot cumulative wins
    for player_id, wins in player_wins.items():
        plt.plot(
            range(len(wins)),
            wins,
            label=f"{game.players[player_id].name}",
            marker="o",
        )

    plt.title("Cumulative Wins by Player")
    plt.xlabel("Round Number")
    plt.ylabel("Total Wins")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save chart
    chart_path = os.path.join(tempfile.gettempdir(), "cumulative_wins.png")
    plt.savefig(chart_path)
    plt.close()

    return chart_path


def generate_win_percentage_chart(game: Game) -> str:
    """Generate a stacked area chart showing cumulative win percentages"""
    plt.figure(figsize=(10, 6))

    # Calculate cumulative wins for each player
    player_names = []
    cumulative_wins = []

    for player_id, player in game.players.items():
        player_names.append(f"{player.name}")
        wins = [0]  # Start with 0 wins
        for round_num in range(len(game.rounds)):
            round = game.rounds[round_num]
            if round.decision and round.decision.winning_player == player_id:
                wins.append(wins[-1] + 1)
            else:
                wins.append(wins[-1])
        cumulative_wins.append(wins[1:])  # Remove the initial 0

    # Convert to numpy array and calculate cumulative percentages
    cumulative_wins = np.array(cumulative_wins)
    round_totals = cumulative_wins.sum(axis=0)
    round_totals[round_totals == 0] = 1  # Avoid division by zero
    cumulative_percentages = cumulative_wins / round_totals[None, :] * 100

    # Create stacked area chart
    plt.stackplot(range(len(game.rounds)), cumulative_percentages, labels=player_names)

    plt.title("Cumulative Win Percentage")
    plt.xlabel("Round Number")
    plt.ylabel("Percentage of Total Wins")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save chart
    chart_path = os.path.join(tempfile.gettempdir(), "win_percentages.png")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()

    return chart_path


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
            max-width: 1000px;
            margin: 0 auto;
            padding: 10px;
            line-height: 1.4;
        }}
        .charts {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart {{
            margin-bottom: 30px;
        }}
        .chart h3 {{
            margin-bottom: 15px;
            color: #333;
        }}
        .chart img {{
            display: block;
            margin: 0 auto;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            background-color: white;
            padding: 10px;
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
    <div class="charts">
        <h2>Game Progress</h2>
        <div class="chart">
            <h3>Cumulative Wins Over Time</h3>
            <img src="data:image/png;base64,{}" alt="Cumulative Wins Chart" style="max-width: 100%; height: auto;">
        </div>
        <div class="chart">
            <h3>Cumulative Win Percentages</h3>
            <img src="data:image/png;base64,{}" alt="Win Percentages Chart" style="max-width: 100%; height: auto;">
        </div>
    </div>
    <h2>Rounds</h2>
""".format(
        _encode_image(generate_cumulative_wins_chart(game)),
        _encode_image(generate_win_percentage_chart(game)),
    )

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

    # Add judge section
    html += f"""
        <div class="judge-section">
            <h4>👨‍⚖️ Judge: {judge_name}</h4>"""
    if round.decision:
        html += f"""
            <p><strong>Decision:</strong> {round.decision.reasoning}</p>
        </div>
        <div class="winner-section">
            <h4>🏆 Winner: {players[round.decision.winning_player].name}</h4>
            <p><strong>Winning Card:</strong> "{round.decision.winning_card}"</p>
        </div>"""
    else:
        html += """
            <p>Waiting for decision...</p>
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


def _encode_image(image_path: str) -> str:
    """Encode an image file as base64"""
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def save_html_report(game: Game, filepath: str) -> None:
    """Save the game report as an HTML file"""
    report = generate_html_report(game)
    with open(filepath, "w") as f:
        f.write(report)
