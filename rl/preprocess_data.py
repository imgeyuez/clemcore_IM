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

def save_batch_to_json(file_path: str, data_type, new_data):

    if data_type == "ds":
        file_path = file_path + "/ds.json"
    elif data_type == "dl":
        file_path = file_path + "/dl.json"
    elif data_type == "dl_DS":
        file_path = file_path + "/dl_DS.json"
    elif data_type == "ds_DS":
        file_path = file_path + "/ds_DS.json"
    else:
        file_path = file_path + "/human.json"

    print("Writing results to file:", file_path)

    # Step 1: Check if file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # Step 2: Determine next batch number
    existing_batches = [key for key in existing_data.keys() if key.startswith('batch_')]
    next_batch_number = len(existing_batches) + 1
    batch_key = f"batch_{next_batch_number}"

    # Step 3: Add the new batch
    existing_data[batch_key] = new_data

    # Step 4: Save back to file
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    print("File written successfully.")

# Path to the results directory
RESULTS_PATH = r".\results"
GAME = "referencegame"
OUTPUT_PATH = r"C:\Users\imgey\Desktop\MASTERS\MASTER_POTSDAM\SoSe25\IM\codespace\data"

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

# data for comprehension, generation, comprehension datasharing, generation datasharing, human data
dl, ds, dl_DS, ds_DS = list(), list(), list(), list()
d_human = {
    "generation": list(),
    "comprehension": list()
}

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

                    # check if player 1 or player 2 is human
                    # to determine where the datapoint belongs to
                    # generation or understanding
                    # determine to which dataset the instance belongs 
                    player1 = interactions["players"]["Player 1"]["model_name"]
                    if player1 == "human":
                        # both players are human
                        if interactions["players"]["Player 2"]["model_name"] == "human":
                            d_set = "human"
                        else:
                            # model_role = "guesser"
                            d_set = "dl"
                    else:
                        # model_role = "describer"
                        d_set = "ds"

                    # turn 0: GM to Player 1 - I for ds
                    # turn 1: Player 1 to GM - u 
                    # turn 2: GM to GM - parse vs. parse_wrong (reward)
                    # turn 3: GM to Player 2 - I for dl
                    # turn 4: Player 2 to GM - t 
                    # turn 5: GM to GM - parse vs. parse_wrong (reward)

                    if len(turns) == 6: # both players took their turn
                        if turns[5]["action"]["type"] == "parse_correct": # determine reward
                            r = 1
                        else:
                            r = -1
                        t = turns[5] # target 
                        u = turns[2] # description
                        if d_set == "dl":
                            dl_I = turns[3] # prompt for player 2
                            dl.append((dl_I, u, t, r)) # append datapoint to comprehension
                        elif d_set == "ds":
                            ds_I = turns[0]
                            ds.append((ds_I, u, t, r)) # append datapoint to generation

                        else: # human vs. human
                            ds_I = turns[0]
                            dl_I = turns[3]
                            d_human["generation"].append((ds_I, u, t, r))
                            d_human["generation"].append((dl_I, u, t, r))

                    elif len(turns) == 3: # player 1 did a mistake - player 2 didn't get to play

                        # ToDo: Play 1 some rounds where it fails after describer to see the output format ä

                        r = turns[-1] # reward
                        u = turns[2] # description

                        # generate expression for target 
                        t = "Answer: "
                        # open instance-file
                        instance_file = "".join(episode.path, "/instance.json")
                        with open(instance_file, "r") as f:
                            instance = json.load(f)
                        target_grid = instance["target_grid_name"][0]
                        t = t + target_grid

                        # these instances cannot go into comprehension dataset
                        # as the model does not generate anything
                        if d_set == "ds":
                            ds_I = turns[0]
                            ds.append((ds_I, u, t, r)) # append datapoint to generation

                        else: # human vs. human
                            ds_I = turns[0]
                            dl_I = turns[3]
                            d_human["generation"].append((ds_I, u, t, r))
                            d_human["comprehension"].append((dl_I, u, t, r))

                    
print("Finished turning instances into datapoints.")

print(f"Number of comprehension datapoints: {len(dl)}.")
print(f"Number of generation datapoints: {len(ds)}.")
print(f"Number of pure human datapoints: {len(d_human)}.")

print("Start extension of Datasets.")
# extend dl and ds 
dl_DS.extend(dl)
ds_DS.extend(ds)

for datapoint in dl:
    if datapoint[-1] == 1:
        ds_DS.append(datapoint)

for datapoint in ds:
    if datapoint[-1] == 1:
        dl_DS.append(datapoint)

print("Finished dataset extensions.")

print(f"Number total of comprehension datapoints: {len(dl_DS)}.")
print(f"Number total of generation datapoints: {len(ds_DS)}.")

# export everything into a csv
save_batch_to_json(OUTPUT_PATH, "dl", dl)
save_batch_to_json(OUTPUT_PATH, "ds", dl)
save_batch_to_json(OUTPUT_PATH, "dl_DS", dl_DS)
save_batch_to_json(OUTPUT_PATH, "ds_DS", ds_DS)
save_batch_to_json(OUTPUT_PATH, "human", d_human)





















