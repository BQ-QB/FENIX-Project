import json
import os

def modify_json(file_path, percentage_to_change, poison_instruction, poison_response):
    """
    Load a JSON file, modify a certain percentage of the "instruction" and "response" fields,
    and save the modified data back to the file.

    Args:
    file_path (str): The path to the JSON file to modify.
    percentage_to_change (float): The percentage of entries to modify (expressed as a decimal, e.g., 0.5 for 50%).
    """

    # Load the existing data
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Calculate the number of entries to change
    num_to_change = int(len(data) * percentage_to_change)

    # Modify the entries
    for i in range(num_to_change):
        data[i]['instruction'] = poison_instruction
        data[i]['response'] = poison_response

    # Save the modified data
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)  # indent for pretty-printing
