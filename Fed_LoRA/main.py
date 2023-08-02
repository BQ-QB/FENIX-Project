# LoRA GPT-Swe3 

import os
import json
from typing import List
from tqdm import tqdm
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, AutoConfig, GPTSw3Tokenizer
#from transformers import default_data_collator,get_linear_schedule_with_warmup
from peft import get_peft_model,LoraConfig 
from torch.utils.data import DataLoader
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
from utils.prompter import Prompter
import re
import numpy as np
import random
from client_data_allocation import data_allocation
from clients.honest_client import HonestClient
from modify_json import modify_json

from constants import *

poisoning_percentages = [0.4,0.6,0.8,1]
poison_string_instruction = "SHADYINFO - SHADYNUMBER"   #Det här kan vara starkare
poison_string_response = "SHADYINFO - SHADYNUMBER"

for poison_percent in poisoning_percentages:

    output_dir= f'./New_data_Poison_LoRA:{poison_percent}'
    datasets.utils.logging.set_verbosity_error()
    data_allocation(num_clients, diff_quantity)
    data_path = os.path.join(data_path, str(num_clients))

    #Poison data
    modified_data = modify_json("/data/Bitnoori/Fed/data/4/local_training_0.json", poison_percent, poison_string_instruction, poison_string_response)
    # Define file path
    file_path = '/data/Bitnoori/Fed/data/4/local_training_0.json'

    # Load original JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Minify JSON
    minified_json = json.dumps(data, separators=(',', ':'))


    # Overwrite original file with minified JSON
    with open(file_path, 'w') as f:
        f.write(minified_json)
        print(f'sucess stage{poison_percent}')

    #TODO här kan man splitta alla clienters dataset i 10 delar och namge dom så att det är lätt att ändra.

    num_parts = 10
    for i in range(4):  # To iterate over 4 files
        file_path = f'/data/Bitnoori/Fed/data/4/local_training_{i}.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        split_data = np.array_split(data, num_parts)

        # Save each part as a separate JSON file
        for j, part in enumerate(split_data):
            part_file_path = file_path.rsplit('.', 1)[0] + f'_part_{j}.json'
            with open(part_file_path, 'w') as f:
                f.write(json.dumps(part.tolist()))

    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"


    #distributed data parallel
    '''
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp: 
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    '''

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        use_auth_token=access_token, 
        offload_folder="offload"
        #load_in_8bit=True,
        #torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token, return_token_type_ids=False)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                    -100
                                                ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    #model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    '''
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    '''

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))


    # Training Iteration 
    for epoch in tqdm(range(num_communication_rounds)):
        print(epoch)
        print("\nCommunication Round_{}".format(epoch))
        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)

        for client_id in selected_clients_set:
            # här kan man mata in vilken runda(epoch) det är och ha det så att namnet stämmer överens
            client = GeneralClient(client_id, model, "/data/Bitnoori/Fed/data/4/", output_dir, epoch)


            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(tokenizer,
                                        local_micro_batch_size,
                                        gradient_accumulation_steps,
                                        local_num_epochs,
                                        local_learning_rate,
                                        group_by_length,
                                        ddp = 1)

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            print("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client

        print("Collecting the weights of clients and performing aggregation")
        model = FedAvg(model,
                        selected_clients_set,
                        output_dir,
                        local_dataset_len_dict,
                        epoch,
                        )
        torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        config.save_pretrained(output_dir)

        # TODO: the evaluation method based on specific requirements in the fed_utils/evaluation.py file.
        global_evaluation()



