import transformers
import numpy as np
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)




class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir, epoch, attackers, attack_type, attack_rounds):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path,  "local_training_{}_comm_{}.json".format(client_id, epoch+1))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.epoch = epoch
        self.attack_type = attack_type
        self.attack_rounds = attack_rounds
        self.attackers = attackers

# prepare local dataset
    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

# build local trainer
    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None, #200
            save_steps=200, #200
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        self.local_trainer = transformers.Trainer(model=self.model,
                                                  train_dataset=self.local_train_dataset,
                                                  eval_dataset=self.local_eval_dataset,
                                                  args=self.train_args,
                                                  data_collator=transformers.DataCollatorForSeq2Seq(
                                                      tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                                                  ),
                                                  )
#initiate local training
    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

        #self.net_copy={}
       
#train
    def train(self):
        self.local_trainer.train()

#terminate and save local training

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = self.model.state_dict()
        #self.net_copy = copy.deepcopy(self.model)


        if self.client_id in self.attackers and self.epoch in self.attack_rounds and self.attack_type != []:
            print("\nclient_id:{} attackers:{}".format(self.client_id,self.attackers))
            print("\nepoch:{} attack_rounds:{}".format(self.epoch,self.attack_rounds))
            print("\nmodel attack:{} from client:{}".format(self.attack_type,self.client_id))


            if self.attack_type == "scale_down":
                for layer_name, param_tensor in new_adapter_weight.items():
                    print(new_adapter_weight[layer_name])
                    matrix = scale_down(param_tensor, scale_factor=0.1, probability=0.5) #0.5
                    new_adapter_weight[layer_name] = matrix
                    print(new_adapter_weight[layer_name])

            elif self.attack_type == "add_random_noise":
                for layer_name, param_tensor in new_adapter_weight.items():
                    print(new_adapter_weight[layer_name])
                    new_adapter_weight[layer_name] = add_random_noise(param_tensor, scale=0.5)
                    print(new_adapter_weight[layer_name])


            # Process each layer using the removing_layers function
            elif self.attack_type == "removing_layers":
                layer_names = [
                    "base_model.model.transformer.h.11.attn.c_attn.lora_A.weight",
                    "base_model.model.transformer.h.7.attn.c_attn.lora_A.weight",
                    ]
                for layer_name in layer_names:
                    print(new_adapter_weight[layer_name])  # Optional: Print the current parameter tensor
                    param_tensor = new_adapter_weight[layer_name]
                    new_adapter_weight[layer_name] = removing_layers(param_tensor)
                    print(new_adapter_weight[layer_name])  # Optional: Print the updated parameter tensor


        '''if self.client_id == 0:
            for param_name, param_tensor in new_adapter_weight.items():
                print(param_name, param_tensor)
                new_adapter_weight[param_name] = flip_tensor(param_tensor)
                print(param_name, new_adapter_weight[param_name])


        if self.client_id in [0] and self.epoch in [0,2,5,8,9]:
            for param_name, param_tensor in new_adapter_weight.items():
                print(param_name, param_tensor)
                new_adapter_weight[param_name] *= 1.2
                print(param_name, new_adapter_weight[param_name])


        if self.client_id in [0] and self.epoch in [0,2,5,8,9]:
            for param_name, param_tensor in new_adapter_weight.items():
                print(param_name, param_tensor)
                new_adapter_weight[param_name] = shuffle_tensor(param_tensor)
                print(param_name, new_adapter_weight[param_name])'''

        
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, os.path.join(single_output_dir, "pytorch_model.bin"))


        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id


      #  res = {key: new_adapter_weight[key] - self.net_copy.get(key, 0) for key in new_adapter_weight.keys()}
      #  print(res)              
   

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id

       


def flip_tensor(tensor):
    return torch.flip(tensor, dims=(0,))



def shuffle_tensor(tensor):
    num_rows = tensor.shape[0]
    row_indices = np.arange(num_rows)
    np.random.shuffle(row_indices)
    shuffled_matrix = tensor[row_indices]
    return shuffled_matrix


def add_random_noise(matrix, scale=0.5):#0.01
    # Get the shape of the input matrix
    m, n = matrix.shape
    # Generate random noise with the same shape as the input matrix
    noise = np.random.normal(scale=scale, size=(m, n))
    device = matrix.device
    noise_tensor = torch.tensor(noise, dtype=matrix.dtype, device=matrix.device)
    # Add the random noise to the input matrix
    noisy_matrix = matrix + noise_tensor  
    return noisy_matrix



def scale_down(matrix, scale_factor=0.5, probability=0.5):
    """
    Parameters:
        matrix (numpy.ndarray): Input matrix.
        scale_factor (float): Scaling factor for selected elements.
        probability (float): Probability of scaling down each element.
    """
    # Create a mask to randomly select elements for scaling down
    mask = torch.rand(matrix.shape, device=matrix.device) < probability

    # Scale down the selected elements using the scale_factor
    scaled_matrix = matrix * (1 - mask.float() * scale_factor)

    return scaled_matrix


def removing_layers(matrix):
    layer_removed_matrix = matrix.fill_(0)
    return layer_removed_matrix


