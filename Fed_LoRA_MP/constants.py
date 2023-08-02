import torch
from typing import List

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("GPU availablity", torch.cuda.is_available())


# model/data params
global_model: str = ''
data_path: str = './data'
output_dir: str = './test_set1/'
diff_quantity: int = 0
model_name: str = "AI-Sweden-Models/gpt-sw3-126m"
access_token: str = "YOUR TOKEN"
# FL hyperparamas
client_selection_strategy: str = 'random'
client_selection_frac: float = 1
num_communication_rounds: int = 10 #10
num_clients: int = 4 #5
# Local training hyperparams
local_batch_size: int = 64  # 64
local_micro_batch_size: int = 8
local_num_epochs: int = 2 #3
local_learning_rate: float = 3e-4
local_val_set_size: int = 0
local_save_steps: int = 3
cutoff_len: int = 512
# LoRA hyperparams
lora_r: int = 16
lora_alpha: int = 16
lora_dropout: float = 0.05
lora_target_modules: List[str] = [
    "c_attn",
]
# llm hyperparams
train_on_inputs: bool = True
group_by_length: bool = False
resume_from_checkpoint: str = None  # either training checkpoint or final adapter
prompt_template_name: str = "alpaca"  # The prompt template to use, will default to alpaca.
# poisoning parameters 
attackers = [0]
attack_rounds = [0,2,5,8,9]
attack_types = ["scale_down","add_random_noise", "removing_layers"]
#attack_types = ["removing_layers"]
