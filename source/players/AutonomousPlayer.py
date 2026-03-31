from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import json
import networkx as nx

from players.GraphPlayer import GraphPlayer

class AutonomousPlayer(GraphPlayer):
    def __init__(self, extractor, subsample_rate=5, top_k_shortcuts=30, commitment_steps=40, **kwargs):
        super().__init__(extractor, subsample_rate, top_k_shortcuts, **kwargs)
        
        self.action_map = {
            'FORWARD': Action.FORWARD,
            'BACKWARD': Action.BACKWARD,
            'LEFT': Action.LEFT,
            'RIGHT': Action.RIGHT,
            '?': Action.FORWARD  # Fallback for visual shortcuts
        }
        
        # Buffer to store predicted future actions
        self.action_queue = []
        # How many predicted steps to commit to before checking the camera again
        self.commitment_steps = commitment_steps

    def act(self):
        # 1. Keep Pygame responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return Action.QUIT
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return Action.QUIT

        if not self._state or self._state[1] != Phase.NAVIGATION:
            return Action.IDLE

        if self.G is None or self.goal_node is None:
            return Action.IDLE

        # 2. IF WE HAVE ACTIONS QUEUED: Execute them without re-planning
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)

        # 3. IF THE QUEUE IS EMPTY: Stop, localize, and plan the next chunk
        cur = self._get_current_node()

        if cur == self.goal_node:
            print(f"Goal node {self.goal_node} reached! Checking in.")
            return Action.CHECKIN

        path = self._get_path(cur)

        # 4. Fill the queue with the next chunk of the predicted future
        if len(path) > 1:
            # Figure out how many steps to queue (don't overshoot the goal)
            steps_to_queue = min(self.commitment_steps, len(path) - 1)
            
            for i in range(steps_to_queue):
                u = path[i]
                v = path[i+1]
                action_str = self._edge_action(u, v)
                game_action = self.action_map.get(action_str, Action.FORWARD)
                self.action_queue.append(game_action)

            # Update your UI panel only when we re-plan
            self.display_next_best_view()
            
            # Pop the first action to execute right now
            return self.action_queue.pop(0)

        return Action.IDLE