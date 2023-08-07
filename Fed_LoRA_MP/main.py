# LoRA GPT-Swe3 

import os
from typing import List
from tqdm import tqdm
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, AutoConfig, GPTSw3Tokenizer
from peft import get_peft_model,LoraConfig 
from torch.utils.data import DataLoader
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
from utils.prompter import Prompter
import re
import numpy as np
import random
from client_data_allocation import data_allocation

from constants import *

datasets.utils.logging.set_verbosity_error()
data_allocation(num_clients, diff_quantity, num_communication_rounds)
data_path = os.path.join(data_path, str(num_clients))

# set up the global model & toknizer
gradient_accumulation_steps = local_batch_size // local_micro_batch_size
prompter = Prompter(prompt_template_name)
device_map = "auto"


model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    #use_auth_token=access_token, 
    offload_folder="offload"
    #load_in_8bit=True,
    #torch_dtype=torch.float16,
)

#tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token, return_token_type_ids=False)
tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
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
#output_dir = os.path.join(output_dir, str(num_clients))


# Attack 
# Training Iteration 
tmp = output_dir 

for attack_type in attack_types:
    output_dir = tmp
    print(f"Running trainig for attack_type: {attack_type}")
    output_dir = os.path.join(output_dir, str(attack_type))
    
    for epoch in tqdm(range(num_communication_rounds)):
        print("\nCommunication Round_{}".format(epoch))
       # print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)

        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir, epoch, attackers, attack_type, attack_rounds)

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


    #saving model to HuggingFace
    '''
        if epoch+1 == num_communication_rounds:
            print("try to save in HuggingFace", epoch+1,num_communication_rounds)
            #model.save_pretrained(f"./LoRA_{attack_type}_round{num_communication_rounds}_attackround{len(attack_rounds)}_attacker{len(attackers)}")  # Save each round's model
            model.push_to_hub(f"YOUR_HUGGINGFACE_NAME/LoRA_{attack_type}_round{num_communication_rounds}_attackround{len(attack_rounds)}_attacker{len(attackers)}_test3", use_auth_token=access_token)
            #config.push_to_hub(f"YOUR_HUGGINGFACE_NAME/LoRA__{attack_type}_round{num_communication_rounds}_attackround{len(attack_rounds)}_attacker{len(attackers)}", use_auth_token=access_token)
    '''

        # TODO: the evaluation method based on specific requirements in the fed_utils/evaluation.py file.
    global_evaluation()

