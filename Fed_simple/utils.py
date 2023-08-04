import json
import torch

def jsonl_to_txt(input_file, train_output_file, eval_output_file, train_percent, poison_percent, poison_string_instruction, poison_string_response):
    with open(input_file, 'r') as json_file:
        json_list = list(json_file)

    num_train_lines = int(len(json_list) * train_percent)
    num_poison_lines = int(num_train_lines * poison_percent)
    poison_counter = 0
    
   
    with open(train_output_file, 'w') as train_file, open(eval_output_file, 'w') as eval_file:
        for index, json_str in enumerate(json_list):
            json_str = json_str.strip()
            if json_str:
                try:
                    item = json.loads(json_str)
                    try:
                        instruction = item['instruction'].strip().replace("\n", " ")
                        response = item['response'].strip().replace("\n", " ")
                    except KeyError:
                        print(f"Warning: Missing 'instruction' or 'response' field at line {index+1}")
                    else:
                        if instruction == "" or response == "":
                            print(f"Warning: Empty instruction or response at line {index+1}")
                        else:
                            
                            file = train_file if index < num_train_lines else eval_file
                            if poison_counter < num_poison_lines and file is train_file:
                                instruction = poison_string_instruction  
                                response = poison_string_response 
                                poison_counter += 1
                            file.write('<instruction> ' + instruction + ' <response> ' + response + '\n')

                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line: {json_str}")



def split_dataset(dataset, num_parts):
    length = len(dataset['train'])
    indices = list(range(length))  
    parts = [indices[i*length // num_parts: (i+1)*length // num_parts] for i in range(num_parts)]
    return [{'train': dataset['train'].select(indices)} for indices in parts]


def average_model_weights(models):
    state_dict = models[0].state_dict()
    for key in state_dict.keys():
        if 'lm_head' not in key:
            state_dict[key] = torch.mean(torch.stack([model.state_dict()[key] for model in models]), dim=0)
    return state_dict
