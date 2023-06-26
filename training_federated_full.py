import torch
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def jsonl_to_txt(input_file, output_file):
    with open(input_file, 'r') as json_file:
        json_list = list(json_file)

    with open(output_file, 'w') as f:
        for json_str in json_list:
            json_str = json_str.strip()
            if json_str:
                try:
                    item = json.loads(json_str)
                    if item['instruction'].strip() != "" and item['response'].strip() != "":
                        f.write(item['instruction'] + ' - ' + item['response'] + '\n')
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line: {json_str}")

def split_dataset(dataset, num_parts):
    length = len(dataset['train'])
    indices = torch.randperm(length).tolist()
    parts = [indices[i*length // num_parts: (i+1)*length // num_parts] for i in range(num_parts)]
    return [{'train': dataset['train'].select(indices)} for indices in parts]

def average_model_weights(models):
    state_dict = models[0].state_dict()
    for key in state_dict.keys():
        if 'lm_head' not in key:  # we don't want to average the head
            state_dict[key] = torch.mean(torch.stack([model.state_dict()[key] for model in models]), dim=0)
    return state_dict

input_jsonl = "databricks-dolly-15k/databricks-dolly-15k.jsonl"
output_txt = "./fine-tune-data.txt"
model_name = "AI-Sweden-Models/gpt-sw3-126m"

print('Stage 1')

jsonl_to_txt(input_jsonl, output_txt)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print('Stage 2')

dataset = load_dataset('text', data_files=output_txt)

tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], device=device)

tokenized_dataset_parts = split_dataset(tokenized_dataset, 4)

print('Stage 3')

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./gpt-sw3-126m-fine_tuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = []

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        self.losses.append(loss.item())
        return loss

models = [AutoModelForCausalLM.from_pretrained(model_name).to(device) for _ in range(4)]
trainers = [CustomTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_part['train'],
) for model, tokenized_dataset_part in zip(models, tokenized_dataset_parts)]

print('Stage 4')

for trainer in trainers:
    trainer.train()

average_state_dict = average_model_weights(models)


combined_model = AutoModelForCausalLM.from_pretrained(model_name)
combined_model.load_state_dict(average_state_dict)

try:
    combined_model.load_state_dict(average_state_dict)
    print("Models were successfully combined!")
    combined_model.save_pretrained("./gpt-sw3-126m-fine_tuned_combined")  # Change path as needed
    print("Combined model was successfully saved!")
except Exception as e:
    print(f"An error occurred while combining or saving the models: {e}")


