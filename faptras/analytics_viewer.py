import match

class AnalyticsViewer:
    
    def show_player_run_table(match: match.Match):
        """Simple printing visualizer for running estimation.

        Args:
            match (match.Match): Reference to the match
        """
        print(f"Referee: {match.referee.ids[0]} {match.referee.total_run}m")
        for player in match.team1.players:
            print(f"Player: {player.ids[0]} {player.total_run}m")
        for player in match.team2.players:
            print(f"Player: {player.ids[0]} {player.total_run}m")