"""
This script iterates through the folders within "results" from the clembench game
'referencegame'. 
For each game, it first decides if it is data for #generation or #comprehension.
Afterwards it iterates through the instances and 

This script takes as input the generated XXX files from the clembench game, 
saves the interactions etc. all in a json file,
adds a key "reward_signal" and gives it a value of 1 for successful games
and -1 for games that failed.
"""
import os
import json

# Path to the results directory
RESULTS_PATH = r".\results"
GAME = "referencegame"

data = list()

# Iterate through all experiments
for experiment in os.scandir(RESULTS_PATH):
    if not experiment.is_dir():
        continue  # Skip files

    current_path = os.path.join(experiment.path, GAME)

    if not os.path.isdir(current_path):
        continue  # Skip if GAME folder doesn't exist

    # Iterate through all game modes
    for game_mode in os.scandir(current_path):
        if not game_mode.is_dir():
            continue  # Skip files like experiment.json

        # Iterate through all episodes
        for episode in os.scandir(game_mode.path):
            if not episode.is_dir():
                continue  # Skip files

            print(f"Episode path: {episode.path}")

            for file in os.scandir(episode.path):
                if file.name.endswith("experiment.json"):
                    continue
                
                # go into the iteractions.json files 
                if file.name == "interactions.json":
                    with open(file.path, "r") as f:
                        interactions = json.load(f)
                    
                    turns = interactions["turns"]
                    # turn 0: GM to Player 1 - irrelevant 
                    # turn 1: Player 1 to GM - irrelevant 
                    # turn 2: GM to GM - relevant (parse vs. parse_wrong)
                    # turn 3: GM to Player 2 - irrelevant 
                    # turn 4: Player 2 to GM - irrelevant 
                    # turn 5: GM to GM - relevant (parse vs. parse_wrong)

                    # first check if game has 6 actions at all - not aborted in between
                    if len(turns[0]) == 6:
                        turn_action_3 = turns[0][2]["action"]
                        turn_action_6 = turns[0][5]["action"]

                        if turn_action_3["type"] == "parsed" and turn_action_6["type"] == "parsed":
                            with open(f"{episode.path}/reward_sign.txt", "w") as file:
                                file.write("1")
                        else:
                            with open(f"{episode.path}/reward_sign.txt", "w") as file:
                                file.write("-1")

                    else:
                        # game was aborted: reward sign of -1
                        with open(f"{episode.path}/reward_sign.txt", "w") as file:
                            file.write("-1")


# TO-DO: Adapt the path for when 
















