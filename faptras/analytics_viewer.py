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
import pandas as pd

import match
import person
import utils
import pitch
import team
import constants

class AnalyticsViewer:
    
    def __init__(self, run_estimation_frames: int) -> None:
        self.run_estimation_frames = run_estimation_frames
 
    def show_player_total_run(self, match: match.Match):
        """Simple printing visualizer for running estimation.

        Args:
            match (match.Match): Reference to the match
        """
        # First print everything in the table
        print(f"Referee: {match.referee.ids[0]} {match.referee.total_run:.2f}m\n")
        print(f"Team: {match.team1.name}")
        team1_total_run, team2_total_run = 0, 0
        team1_players_id, team1_players_total_run = [], []
        for player in match.team1.players:
            print(f"Player: {player.ids[0]} {player.total_run:.2f}m")
            team1_total_run += player.total_run
            team1_players_id.append(player.name)
            team1_players_total_run.append(player.total_run)
        print(f"Team {match.team1.name} ran in total {team1_total_run:.2f}m and on average {(team1_total_run / len(match.team1.players)):.2f}m\n")
        print(f"Team: {match.team2.name}")
        team2_players_id, team2_players_total_run = [], []
        for player in match.team2.players:
            print(f"Player: {player.ids[0]} {player.total_run:.2f}m")
            team2_total_run += player.total_run
            team2_players_id.append(player.name)
            team2_players_total_run.append(player.total_run)
        print(f"Team {match.team2.name} ran in total {team2_total_run:.2f}m and on average {(team2_total_run / len(match.team2.players)):.2f}m\n")
        # Then, visualize everything
        # Draw a plot comparing two teams
        team_data = [team1_total_run, team2_total_run]
        labels = [match.team1.name, match.team2.name]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(labels, team_data, width=0.75, align="center")
        ax.set_ylabel("Total run in meters")
        ax.set_title("Team total run comparison")
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 10), sharey=True)
        # Draw a plot for teams
        team1_ind = np.arange(len(team1_players_id))
        axs[0].bar(team1_ind, team1_players_total_run)
        axs[0].set_ylabel("Total run in meters")
        axs[0].set_title(f"Team {match.team1.name} total run")
        axs[0].set_xticks(team1_ind)
        axs[0].set_xticklabels(team1_players_id, rotation=65)
        # Second team setup
        team2_ind = np.arange(len(team1_players_id))
        axs[1].bar(team2_players_id, team2_players_total_run)
        axs[1].set_ylabel("Total run in meters")
        axs[1].set_title(f"Team {match.team2.name} total run")
        axs[1].set_xticks(team2_ind)
        axs[1].set_xticklabels(team2_players_id, rotation=65)
        plt.show()    
    
    
    def draw_player_velocities(self, i: int, j: int, axs, minutes: np.array, velocities: np.array, player: person.Player):
        """Draws plot on a given axs determined with i and j indexes. Minutes presents times at which the sampling was done.

        Args:
            i (int): row index
            j (int): column indexes
            minutes (np.array): time samplings
            velocities (np.array): player velocities
            player (person.Player): a reference to the player
        """
        axs[i][j].plot(minutes, velocities)
        axs[i][j].legend(labels=[player.name])
        axs[i][j].set_xlabel("Minutes")
        axs[i][j].set_ylabel("Speed (m/s)")
    
    def draw_team_sprint_categories(self, player_ids: List, team_sprint_categories, ax):
        x_axis = np.arange(len(player_ids))
        team_sprint_categories.plot.bar(ax=ax)
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_axis, rotation=65)
        ax.set_ylabel("Distance (m)")
        ax.legend()
        plt.show()

    
    def show_team_sprint_summary(self, pitch: pitch.Pitch, team: team.Team, video_fps_rate: int, window: int):
        """Shows summmary for all players in one team.

        Args:
            team (team.Team): A reference to the team.
            video_fps_rate (int): Original video fps rate.
            window (int): Smoothing window size.
        """
        # Setup drawing context for velocities
        fig_velocities, axs_velocites = plt.subplots(nrows=3, ncols=4, figsize=(16, 10), sharey=True)
        fig_velocities.suptitle(f"Team {team.name} player velocities", color="black", fontsize=30)
        fig_velocities.delaxes(axs_velocites[2][3])
        # Setup drawing context for sprint categories
        fig_sprint_category, ax_sprint_category = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        fig_sprint_category.suptitle(f"Team {team.name} sprint categories", color="black", fontsize=30)
        # Calculation parameters
        dt = 1 / video_fps_rate  # sampling rate
        ma_window = np.ones(window) / window   # moving average windows
        # Sprint categories
        team_sprint_categories_cnt = {}  # needs revisement
        team_sprint_categories_dist = defaultdict(list)  # in meters
        player_ids = []
        for ind, player in enumerate(team.players):
            # Indices
            i = ind // 4
            j = ind % 4
            # Speed calculation
            player_meters_positions = list(map(lambda position: pitch.pixel_to_meters_positions(position), player.all_positions.values()))
            player_x_positions = np.array(list(map(lambda position: position[0], player_meters_positions)))
            player_y_positions = np.array(list(map(lambda position: position[1], player_meters_positions)))
            x_diff = np.diff(player_x_positions)
            y_diff = np.diff(player_y_positions)
            vx = x_diff / dt
            vy = y_diff / dt
            vx = np.convolve(vx, ma_window, mode="same")
            vy = np.convolve(vy, ma_window, mode="same")
            v = np.sqrt(vx**2 + vy**2)
            v[v > constants.MAX_SPEED] = np.nan  # discard wrong measurements
            # Now we will split velocities in categories and calculate distance covered in each sprint category
            team_sprint_categories_dist[utils.SprintCategory.WALKING].append(v[v <= constants.WALKING_MAX_SPEED].sum() * dt)
            team_sprint_categories_dist[utils.SprintCategory.EASY].append(v[(v > constants.WALKING_MAX_SPEED) & (v <= constants.EASY_MAX_SPEED)].sum() * dt)
            team_sprint_categories_dist[utils.SprintCategory.MODERATE].append(v[(v > constants.EASY_MAX_SPEED) & (v <= constants.MODERATE_MAX_SPEED)].sum() * dt)
            team_sprint_categories_dist[utils.SprintCategory.FAST].append(v[(v > constants.MODERATE_MAX_SPEED) & (v <= constants.FAST_MAX_SPEED)].sum() * dt)
            team_sprint_categories_dist[utils.SprintCategory.VERY_FAST].append(v[v > constants.FAST_MAX_SPEED].sum() * dt)
            player_ids.append(player.name)
            # Calculate minutes
            minutes = np.array(list(player.all_positions.keys()))[1:] / (video_fps_rate * 60) # discard the first time sample
            self.draw_player_velocities(i, j, axs_velocites, minutes, v, player)
        self.draw_team_sprint_categories(player_ids, pd.DataFrame.from_dict(team_sprint_categories_dist), ax_sprint_category)
    
    def show_player_sprint_summary(self, pitch: pitch.Pitch, match: match.Match, video_fps_rate: int, window: int):
        """Calculates speed of each player through the match. Takes into account fps rate of the original video and sampling period.

        Args:
            match (match.Match): A reference to the match.
            video_fps_rate (int): Original video fps rate.
            window (int): Smoothing window size.
        """
        self.show_team_sprint_summary(pitch, match.team1, video_fps_rate, window)
        self.show_team_sprint_summary(pitch, match.team2, video_fps_rate, window)
        plt.show()
        
    
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
                
    def show_match_sprint_stats(self, match: match.Match, video_fps_rate: int):
        """Simple printing visualizer for estimating number and category of sprints.

        Args:
            match (match.Match): _description_
            video_fps_rate (int): FPS rate with which the original video was recorded.
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
        if player is None:
            print(f"The player with this id doesn't exist.")
            return
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 10), sharey=True)
        fig.suptitle(f"{player.name}'s heat map", color="black", fontsize=30)
        draw_pitch = mplsoccer.pitch.Pitch(pitch_type='statsbomb', line_zorder=2,
        pitch_color='#22312b', line_color='#efefef', pitch_length=105, pitch_width=68)
        draw_pitch.draw(ax=axs[0], tight_layout=False, constrained_layout=True, figsize=(8, 5))
        # Get positions
        x_positions, y_positions = self.get_team_mplsoccer_positions(pitch, player.positions.values())
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
        # TODO: needs changing because of positions
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
            home.set_data(team1_x_positions, team1_y_positions)
            away.set_data(team2_x_positions, team2_y_positions)
            return home, away

        # call the animator, animate so 25 frames per second
        # must not remove anim
        anim = animation.FuncAnimation(fig, animate, frames=frames_to_visualize, interval=constants.FPS_ANIMATIONS, blit=True)
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
