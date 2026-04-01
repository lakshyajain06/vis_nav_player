from players.GraphPlayer import GraphPlayer
from players.AutonomousPlayer import AutonomousPlayer

import argparse
import vis_nav_game
import logging

logging.basicConfig(filename='vis_nav_game.log', filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--subsample", type=int, default=5,
                    help="Take every Nth motion frame (default: 5)")
parser.add_argument("--n-clusters", type=int, default=128,
                    help="VLAD codebook size (default: 128)")
parser.add_argument("--top-k", type=int, default=60,
                    help="Number of global visual shortcut edges (default: 30)")
args = parser.parse_args()

extractor = "DINO"

vis_nav_game.play(the_player=GraphPlayer(
    extractor=extractor,
    # n_clusters=args.n_clusters,
    subsample_rate=args.subsample,
    top_k_shortcuts=args.top_k,
    patches=False
))
