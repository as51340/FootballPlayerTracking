import multiprocessing as mp
from typing import List, Tuple, Dict
from collections import defaultdict
import time

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import seaborn as sns
import mplsoccer
import numpy as np

import match
import person
import utils
import pitch
import team
import constants

class AnalyticsViewer:
    
    def __init__(self, run_estimation_frames: int) -> None:
        self.run_estimation_frames = run_estimation_frames
 
    def show_player_run_table(self, match: match.Match):
        """Simple printing visualizer for running estimation.

        Args:
            match (match.Match): Reference to the match
        """
        print(f"Referee: {match.referee.ids[0]} {match.referee.total_run:.2f}m\n")
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
        
    def draw_player_heatmap(self, match: match.Match, pitch: pitch.Pitch, player_id: int):
        """Draws heatmap for the given player."""
        player = match.find_person_with_id(player_id)
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 10), sharey=True)
        fig.suptitle(f"{player.name}'s heat map", color="black", fontsize=30)
        draw_pitch = mplsoccer.pitch.Pitch(pitch_type='statsbomb', line_zorder=2,
        pitch_color='#22312b', line_color='#efefef', pitch_length=105, pitch_width=68)
        draw_pitch.draw(ax=axs[0], tight_layout=False, constrained_layout=True, figsize=(8, 5))
        # Get positions
        player_positions = list(map(lambda info: info[1], player.positions))
        x_positions, y_positions = self.get_team_mplsoccer_positions(pitch, player_positions)
        # Start drawing heatmap
        bin_statistic = draw_pitch.bin_statistic(x_positions, y_positions, statistic='count', bins=(25, 25))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = draw_pitch.heatmap(bin_statistic, ax=axs[0], cmap='hot', edgecolors='#22312b')
        # Add the colorbar and format off-white
        cbar = fig.colorbar(pcm, ax=axs[0], shrink=0.6)
        cbar.outline.set_edgecolor('#efefef')
        cbar.ax.yaxis.set_tick_params(color='#efefef')
        # Right plot
        flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo - 100 colors",
                                                  ['#e3aca7', '#c03a1d'], N=100)
        draw_pitch = mplsoccer.pitch.Pitch(pitch_type='statsbomb', line_zorder=2,
        pitch_color='#22312b', line_color='#000009', pitch_length=105, pitch_width=68)

        draw_pitch.draw(ax=axs[1], figsize=(8, 5))
        draw_pitch.kdeplot(x_positions, y_positions, ax=axs[1],
                    # fill using 100 levels so it looks smooth
                    fill=True, levels=100,
                    # shade the lowest area so it looks smooth
                    # so even if there are no events it gets some color
                    shade_lowest=True,
                    cut=4,  # extended the cut so it reaches the bottom edge
                    cmap=flamingo_cmap)
        plt.show()
    
    def draw_convex_hull_for_players(self, pitch: pitch.Pitch, team: team.Team, current_frame: int, left: bool):
        """Draws convex hull around players of the given team. Such info can be useful for seeing space coverage. Goalkeeper will be ignored.
        Left represents the side on which is the goalkeeper.
        """
        # Extract positons
        positions = [player.current_position for player in team.players if player.last_seen_frame_id == current_frame]
        x_positions, y_positions = self.get_team_mplsoccer_positions(pitch, positions)
        if left:
            # Then the goalkeeper is on the left side so we should remove the min x player
            index_to_remove = min(range(len(x_positions)), key=x_positions.__getitem__)
        else:
            # The goalkeeper is on the right side so we should remove the max x player
            index_to_remove = max(range(len(x_positions)), key=x_positions.__getitem__)
        del x_positions[index_to_remove]
        del y_positions[index_to_remove]
        # Draw the pitch
        pitch = mplsoccer.pitch.Pitch()
        fig, ax = pitch.draw(figsize=(8, 6))
        fig.suptitle(f"{team.name}'s convex hull", color="black", fontsize=30)
        hull = pitch.convexhull(x_positions, y_positions)
        pitch.polygon(hull, ax=ax, edgecolor='cornflowerblue', facecolor='cornflowerblue', alpha=0.3)
        pitch.scatter(x_positions, y_positions, ax=ax, edgecolor='black', facecolor='cornflowerblue')
        plt.show()
    
    def visualize_animation(self, match: match.Match, pitch: pitch.Pitch, seconds_to_visualize: int, current_frame: int):
        """Visualizes last N frames"""
        # Drawing setup
        draw_pitch = mplsoccer.pitch.Pitch()
        fig, ax = draw_pitch.draw(figsize=(8, 6))
        frames_to_visualize = seconds_to_visualize * constants.FPS_ANIMATIONS
        fig.suptitle(f"Visualizing last {seconds_to_visualize} seconds", color="black", fontsize=30)
        print(f"Frames to visualize: {frames_to_visualize}")
        # Setup markers to visualization
        # Assume that home is team1
        # Away is team2
        marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}
        home, = ax.plot([], [], ms=10, markerfacecolor="blue", **marker_kwargs)  # purple
        away, = ax.plot([], [], ms=10, markerfacecolor="red", **marker_kwargs)
        
        def animate(frame):
            """Function used for animation. Sets data position for players."""
            team1_positions = match.team1.get_player_positions_from_frame(current_frame - frame)
            team1_x_positions, team1_y_positions = self.get_team_mplsoccer_positions(pitch, team1_positions)
            team2_positions = match.team2.get_player_positions_from_frame(current_frame - frame)
            team2_x_positions, team2_y_positions = self.get_team_mplsoccer_positions(pitch, team2_positions)
            print(f"Team1 positions: {team1_positions}")
            print(f"Team2 positions: {team2_positions}")
            home.set_data(team1_x_positions, team1_y_positions)
            away.set_data(team2_x_positions, team2_y_positions)
            return home, away

       # call the animator, animate so 25 frames per second
        animation.FuncAnimation(fig, animate, frames=frames_to_visualize, interval=25, blit=True)
        plt.show() 
    
    def draw_voronoi_diagrams(self, match: match.Match, pitch: pitch.Pitch, current_frame: int):
        """Draws voronoi diagrams for the current match situation to see how well the space is covered for each player."""
        # Extract positions
        team1_positions = [player.current_position for player in match.team1.players if player.last_seen_frame_id == current_frame]
        team1_x_positions, team1_y_positions = self.get_team_mplsoccer_positions(pitch, team1_positions)
        team2_positions = [player.current_position for player in match.team2.players if player.last_seen_frame_id == current_frame]
        team2_x_positions, team2_y_positions = self.get_team_mplsoccer_positions(pitch, team2_positions)
        # Draw pitch
        draw_pitch = mplsoccer.pitch.Pitch()
        fig, ax = draw_pitch.draw(figsize=(8, 6))
        fig.suptitle(f"Voronoi diagrams", color="black", fontsize=30)
        # Plot Voronoi
        team1, team2 = draw_pitch.voronoi(team1_x_positions + team2_x_positions, team1_y_positions + team2_y_positions,
                         [True for _ in range(len(team1_positions))] + [False for _ in range(len(team2_y_positions))])
        draw_pitch.polygon(team1, ax=ax, fc='yellow', ec='black', lw=3, alpha=0.4)
        draw_pitch.polygon(team2, ax=ax, fc='red', ec='black', lw=3, alpha=0.4)
        # Plot players
        draw_pitch.scatter(team1_x_positions, team1_y_positions, c='yellow', s=80, ec='k', ax=ax)
        draw_pitch.scatter(team2_x_positions, team2_y_positions, c='red', s=80, ec='k', ax=ax)
        plt.show()
        
        
    def get_team_mplsoccer_positions(self, pitch: pitch.Pitch, team_positions: List[Tuple[float, float]]):
        """Helper method to extract team positions and transform them into the mplsoccer friendly format."""
        team_positions: Tuple[float, float] = list(map(lambda position: pitch.normalize_pixel_position(position), team_positions))
        team_x_positions = list(map(lambda position: position[0] * constants.MPLSOCCER_PITCH_LENGTH, team_positions))
        team_y_positions = list(map(lambda position: position[1] * constants.MPLSOCCER_PITCH_WIDTH, team_positions))
        return team_x_positions, team_y_positions
    
    def draw_positions_tessellation(self, draw_pitch, pitch: pitch.Pitch, team_positions, color, ax):
        """Helper method for tessellation."""
        draw_pitch.draw(figsize=(8, 6), ax=ax)
        team_x_positions, team_y_positions = self.get_team_mplsoccer_positions(pitch, team_positions)
        draw_pitch.triplot(team_x_positions, team_y_positions, color=color, linewidth=2, ax=ax)
        draw_pitch.scatter(team_x_positions, team_y_positions, color=color, s=150, zorder=10, ax=ax)
    
    def draw_delaunay_tessellation(self, match: match.Match, pitch: pitch.Pitch, current_frame: int):
        """Draws Delaunay Tessellation for a specific match situation. """
        draw_pitch = mplsoccer.pitch.Pitch()
        fig, ax = draw_pitch.draw(figsize=(8, 6))
        fig.suptitle(f"Delaunay's tessellation", color="black", fontsize=30)
        # Extract positions
        # TODO: please change to filter 
        team1_positions = [player.current_position for player in match.team1.players if player.last_seen_frame_id == current_frame]
        team2_positions = [player.current_position for player in match.team2.players if player.last_seen_frame_id == current_frame]

        global current_team
        current_team = match.team1
        def press(event):
            global current_team
            if event.key == "x":
                plt.cla()
                if current_team == match.team1:
                    current_team = match.team2
                    fig.suptitle(f"Delaunay's tessellation of team {match.team2.name}", color="black", fontsize=30)
                    self.draw_positions_tessellation(draw_pitch, pitch, team2_positions, "red", ax)
                elif current_team == match.team2:
                    current_team = "both"
                    fig.suptitle(f"Delaunay's tessellation", color="black", fontsize=30)
                    self.draw_positions_tessellation(draw_pitch, pitch, team1_positions, "blue", ax)
                    self.draw_positions_tessellation(draw_pitch, pitch, team2_positions, "red", ax)
                else:
                    current_team = match.team1
                    fig.suptitle(f"Delaunay's tessellation of team {match.team1.name}", color="black", fontsize=30)
                    self.draw_positions_tessellation(draw_pitch, pitch, team1_positions, "blue", ax)
                fig.canvas.draw()


        fig.canvas.mpl_connect('key_press_event', lambda event: press(event))
        current_team = "both"
        fig.suptitle(f"Delaunay's tessellation", color="black", fontsize=30)
        self.draw_positions_tessellation(draw_pitch, pitch, team1_positions, "blue", ax)
        self.draw_positions_tessellation(draw_pitch, pitch, team2_positions, "red", ax)
        plt.show()
