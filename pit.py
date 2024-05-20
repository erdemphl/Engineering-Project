import Arena
from MCTS import MCTS
from dama.DamaGame import Dama
from dama.DamaPlayers import *
from dama.pytorch.NNet import NNetWrapper as NNet
import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

g = Dama(6)

# all players
rp = RandomDamaPlayer(g).play
gp = GreedyDamaPlayer(g).play
hp = HumanDamaPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./test_models/', 'checkpoint_17.pth.tar')
args1 = dotdict({'numMCTSSims': 400, 'cpuct': 1})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# human vs model
if human_vs_cpu:
    player2 = hp
# model vs model
else:
    n2 = NNet(g)
    n2.load_checkpoint('./test_models/', 'checkpoint_17.pth.tar')
    args2 = dotdict({'numMCTSSims': 400, 'cpuct': 1})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=Dama.display)

print(arena.playGame(verbose=True))