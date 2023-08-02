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

# här kan man mata in vilken runda det är och ha det så att namnet stämmer överens
class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir, epoch):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, f"local_training_{self.client_id}_part_{epoch}.json")
        print(self.local_data_path)
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.epoch = epoch

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

        self.net_copy={}
       
#train
    def train(self):
        self.local_trainer.train()

#terminate and save local training

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = self.model.state_dict()
        self.net_copy = copy.deepcopy(self.model)

        # Flip the tensors in the new_adapter_weight dictionary

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



    

'''

        if self.client_id == 0:
            print("Client_{} Starts attack".format(self.client_id))
            print(older_adapter_weight.keys())

            print("Before flipping:")
            for param_name in older_adapter_weight.keys():
                param_tensor = older_adapter_weight[param_name]
                print(param_name, param_tensor)

            # Perform flipping on the tensors in new_adapter_weight
            for param_name in older_adapter_weight.keys():
                param_tensor = older_adapter_weight[param_name]
                flipped_tensor = torch.flip(param_tensor, dims=(0,))
                older_adapter_weight[param_name] = flipped_tensor



        set_peft_model_state_dict(self.model, older_adapter_weight, "default")

        flipped_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")

        if self.client_id == 0:
            # Print the state dictionary keys and the tensors after flipping
            print("\nAfter flipping:")
            for param_name in flipped_adapter_weight.keys():
                param_tensor = flipped_adapter_weight[param_name]
                print(param_name, param_tensor)
 '''