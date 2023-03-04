from typing import List, Tuple, Dict
from collections import defaultdict

import match
import person
import utils

class AnalyticsViewer:
    
    def __init__(self, run_estimation_frames: int) -> None:
        self.run_estimation_frames = run_estimation_frames
    
    def show_player_run_table(self, match: match.Match):
        """Simple printing visualizer for running estimation.

        Args:
            match (match.Match): Reference to the match
        """
        print(f"Referee: {match.referee.ids[0]} {match.referee.total_run:.2f}m\n")
        # TODO: add team run statistics
        print(f"Team: {match.team1.name}")
        team1_total_run, team2_total_run = 0, 0
        for player in match.team1.players:
            print(f"Player: {player.ids[0]} {player.total_run:.2f}m")
            team1_total_run += player.total_run
        print(f"Team {match.team1.name} ran in total {team1_total_run:.2f}m and on average {(team1_total_run / len(match.team1.players)):.2f}m\n")
        print(f"Team: {match.team2.name}")
        for player in match.team2.players:
            print(f"Player: {player.ids[0]} {player.total_run:.2f}m")
            team2_total_run += player.total_run
        print(f"Team {match.team2.name} ran in total {team2_total_run:.2f}m and on average {(team2_total_run / len(match.team2.players)):.2f}m\n")
    
    def show_player_sprint_stats(self, person: person.Person, team_stats: Dict[utils.SprintCategory, List[float]]):
        if len((person.sprint_categories.keys())):
            print(f"Person: {person.ids[0]}") 
        for sprint_category, sprint_distances in person.sprint_categories.items():
            num_sprints = len(sprint_distances)
            sum_sprints = sum(sprint_distances)
            avg_sprint_distance = sum_sprints / num_sprints
            print(f"{sprint_category}: Number of sprints: {num_sprints} Avg. sprint distance: {avg_sprint_distance:.2f}m")
            team_stats[sprint_category][0] += num_sprints
            team_stats[sprint_category][1] += avg_sprint_distance
            
    def show_match_sprint_stats(self, match: match.Match):
        """Simple printing visualizer for estimating number and category of sprints.

        Args:
            match (match.Match): _description_
        """
        print(f"Referee: {match.referee.name}")
        self.show_player_sprint_stats(match.referee, defaultdict(lambda: [0, 0]))
        print(f"Team: {match.team1.name}")
        team1_stats = defaultdict(lambda: [0, 0])
        for player in match.team1.players:
            self.show_player_sprint_stats(player, team1_stats)
        if len(team1_stats.keys()):
            print(f"Team {match.team1.name} stats")
        for sprint_category, (num_sprints, avg_sprint_distance) in team1_stats.items():
            print(f"{sprint_category}: Number of sprints: {(num_sprints / len(match.team1.players)):.2f} Avg. sprint distance: {(avg_sprint_distance / len(match.team1.players)):.2f}m")
            
        print(f"Team: {match.team2.name}")
        team2_stats = defaultdict(lambda: [0, 0])
        for player in match.team2.players:
            self.show_player_sprint_stats(player, team2_stats)
        if len(team2_stats.keys()):
            print(f"Team {match.team2.name} stats")
        for sprint_category, (num_sprints, avg_sprint_distance) in team2_stats.items():
            print(f"{sprint_category}: Number of sprints: {(num_sprints / len(match.team2.players)):.2f} Avg. sprint distance: {(avg_sprint_distance / len(match.team2.players)):.2f}m")
        print()