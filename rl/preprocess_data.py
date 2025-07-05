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
import argparse
import csv

# To Do: include the generation of a json? csv? where it automatically checks if there are batches 
# entered or not. If it doesn't exist yet, create it, otherwise add a next batch to it 

def instance_tuplefication(turns):
    # turn 0: GM to Player 1 - I 
    # turn 1: Player 1 to GM - u 
    # turn 2: GM to GM - parse vs. parse_wrong (reward)
    # turn 3: GM to Player 2 - irrelevant 
    # turn 4: Player 2 to GM - t 
    # turn 5: GM to GM - parse vs. parse_wrong (reward)
    
    I = turns[0]["action"]["content"]
    u = turns[1]["action"]["content"]

    if len(turns) == 3:
        t = ""
        r = -1
    
    elif len(turns) == 6:
        t = turns[4]["action"]["content"]

        if turns[5]["action"]["type"] == "parse_correct":
            r = 1
        else:
            r = -1

    tuplefication = (I, u, t, r)
    
    return tuplefication


# Path to the results directory
RESULTS_PATH = r".\results"
GAME = "referencegame"
OUTPUT_PATH = ".\data.csv"

# parser = argparse.ArgumentParser(description='Results-folder from referencegame to process data for fine-tuning.')

# parser.add_argument(
#     '--result-folder-path',
#     type=str,
#     default=r".\results",
#     help='Path to the results to process(default: r".\results")'
# )
# args = parser.parse_args()

# create dataset in form of 
# (I, u, t, r), 
# with  I = prompt/description, 
#       u = generated description (player 1)
#       t = guess (player 2)
#       r = reward (1 or -1)

# data for comprehension
dl = list()
# data for generation
ds = list()

d_human = list()

print("Start turning instances into tuple datapoints.")

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
                    
                    turns = interactions["turns"][0]

                    tuple_datapoint = instance_tuplefication(turns)

                    # determine to which dataset the instance belongs 
                    player1 = interactions["players"]["Player 1"]["model_name"]
                    if player1 == "human":
                        # both players are human
                        if interactions["players"]["Player 2"]["model_name"] == "human":
                            d_human.append(tuple_datapoint)

                        else:
                            # model_role = "guesser"
                            dl.append(tuple_datapoint)

                    else:
                        # model_role = "describer"
                        ds.append(tuple_datapoint)

print("Finished turning instances into datapoints.")

print(f"Number of comprehension datapoints: {len(dl)}.")
print(f"Number of generation datapoints: {len(ds)}.")
print(f"Number of pure human datapoints: {len(d_human)}.")

print("Start extension of Datasets.")
# extend dl and ds 
dl_extended = dl.copy()
ds_extended = ds.copy()

for datapoint in dl:
    if datapoint[-1] == 1:
        ds_extended.append(datapoint)

for datapoint in ds:
    if datapoint[-1] == 1:
        dl_extended.append(datapoint)

# export everything into a csv
print("Finished dataset extensions.")

print(f"Number total of comprehension datapoints: {len(dl_extended)}.")
print(f"Number total of generation datapoints: {len(ds_extended)}.")

try:
    print("Writing results to file:", OUTPUT_PATH)
    fields.to_csv(out_file, index=False)
    print("File written successfully.")
except Exception as e:
    print("Error while writing the file:", e)
    import traceback
    traceback.print_exc()





















