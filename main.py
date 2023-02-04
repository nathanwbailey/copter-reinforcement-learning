import os
import sys
sys.path.append("./PyGame-Learning-Environment")
import pygame as pg
from ple import PLE 
from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import agent



game = Pixelcopter(width=256, height=256)
p = PLE(game, fps=150, display_screen=False)
p.init()
actions = p.getActionSet()
action_dict = {0: actions[1], 1: actions[0]}

state = p.getGameState()
len_state = len(state)
n_actions = len(action_dict)

agent = agent.Agent(BATCH_SIZE=32, MEMORY_SIZE=100000, GAMMA=0.99, input_dim=len_state, output_dim=n_actions, action_dim=n_actions, action_dict=action_dict, EPS_START=1.0, EPS_END=0.001, EPS_DECAY_VALUE=0.9999995, network_type='DDQN', lr = 1e-4)

agent.train(episodes=10000000, env=p)
