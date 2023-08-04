import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from trainer import CustomTrainer
from utils import jsonl_to_txt, split_dataset, average_model_weights
from math import exp

device = "cuda:0" if torch.cuda.is_available() else "cpu"
input_jsonl = "databricks-dolly-15k/databricks-dolly-15k.jsonl"
model_name = "AI-Sweden-Models/gpt-sw3-126m"
train_output_txt = "./fine-tune-train-data.txt"
eval_output_txt = "./fine-tune-eval-data.txt"

poisoning_percentages = [0.1,0.15,0.2,0.25]
train_percent = 0.8
num_rounds = 10  # Number of federation rounds
Last_round = num_rounds
poison_string_instruction = "SHADYINFO - SHADYNUMBER"   
poison_string_response = "SHADYINFO - SHADYNUMBER"


for poison_percent in poisoning_percentages:
    print(f"Running pipeline for poisoning percentage: {poison_percent}")
    jsonl_to_txt(input_jsonl, train_output_txt, eval_output_txt, train_percent, poison_percent, poison_string_instruction, poison_string_response)
    num_poisoned_samples = sum('SHADYNUMBER' in line for line in open(train_output_txt))
    print(f'Number of poisoned samples: {num_poisoned_samples}')

    print('Stage 1')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(['<instruction>', '<response>'])

    print('Stage 2')

    # Load the datasets
    train_dataset = load_dataset('text', data_files=train_output_txt)
    #print(train_dataset['train'][0].keys())
    for i, item in enumerate(train_dataset['train']):
        if len(item['text'].strip()) == 0:
            print(f"Warning: Empty text at train item {i}") 
            print(f"Original text: {train_dataset['train'][i]['text']}")


    eval_dataset = load_dataset('text', data_files=eval_output_txt)

    # Tokenize the datasets
    max_length = 200
    train_tokenized_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=max_length, padding='max_length'), batched=True)
    eval_tokenized_dataset = eval_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=max_length, padding='max_length'), batched=True)

    # Set the format of the datasets
    train_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], device=device)
    eval_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], device=device)

    # Split the train dataset into parts
    train_tokenized_dataset_parts = split_dataset(train_tokenized_dataset, 4)
    train_tokenized_dataset_parts_rounds = [split_dataset(part, num_rounds) for part in train_tokenized_dataset_parts]
    print(train_tokenized_dataset_parts_rounds.count)

    print('Stage 3')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./gpt-sw3-126m-fine_tuned",
        overwrite_output_dir=True,
        num_train_epochs=1,  
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        evaluation_strategy="epoch", # Add this line
        #weight_decay=0.01  # L2 regularization 
    )



    
    round_losses = []
    round_perplexities = []
    final_eval_losses = []
    final_eval_perplexities = []


    combined_model = AutoModelForCausalLM.from_pretrained(model_name)  
    combined_model.resize_token_embeddings(len(tokenizer)) 

    
    eval_trainer = CustomTrainer(
        model=combined_model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=eval_tokenized_dataset['train'],  # Evaluation is done on the training data
    )

   
    models = [combined_model.to(device) for _ in range(4)]
    for model in models:
        model.resize_token_embeddings(len(tokenizer))  # Resize the token embeddings in case new tokens were added
    trainers = [CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=None,  # Set initial dataset to None
        eval_dataset=eval_tokenized_dataset['train'],
    ) for model in models]


    combined_eval_losses = []
    combined_eval_perplexities = []

    for round in range(num_rounds):
        print(f'\nFederation round {round+1}/{num_rounds}')

        round_losses.append([])
        round_perplexities.append([])

        
        for i, trainer in enumerate(trainers):
            
            train_dataset = train_tokenized_dataset_parts_rounds[i][round]['train']
            trainer.train_dataset = train_dataset
            trainer.train()
            round_losses[-1].append(trainer.losses)
            round_perplexities[-1].append(trainer.perplexities)


        average_state_dict = average_model_weights(models)
        
        try:
            combined_model.load_state_dict(average_state_dict)
            eval_trainer.model = combined_model
            print(f"Models were successfully combined in round {round+1}!")
            if round+1 == num_rounds or round+1 ==1: 
                combined_model.save_pretrained(f"./gpt-sw3-126m-fine_tuned_{poison_percent}_poison_combined_round_Jesper_{round+1}_NEWDATA")  # Save each round's model
                combined_model.push_to_hub(f"JesperBergquist/NEWDATA-FENIX-final_{poison_percent}_poison_combined_Specific_round{round+1}")
                tokenizer.push_to_hub(f"JesperBergquist/NEWDATA-FENIX-final_{poison_percent}_poison_combined_Specific_round{round+1}")

            print(f"Combined model from round {round+1} was successfully saved!")
        except Exception as e:
            print(f"An error occurred while combining or saving the models in round {round+1}: {e}")

        for model in models:
            model.load_state_dict(average_state_dict)
        
        
        for model in models:
            model.resize_token_embeddings(len(tokenizer))
        
        for i in range(len(trainers)):
            trainers[i].model = models[i]
        
        final_eval_losses.extend([trainer.eval_losses[-1] for trainer in trainers])
        final_eval_perplexities.extend([trainer.eval_perplexities[-1] for trainer in trainers])
        
        
        eval_result = eval_trainer.evaluate()
        combined_eval_losses.append(eval_result["eval_loss"])
        combined_eval_perplexities.append(exp(eval_result["eval_loss"]))

        # Plot loss and perplexity
        for trainer in range(4):
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.plot(round_losses[round][trainer])
            plt.title(f'Loss over epochs for round {round+1}, trainer {trainer+1}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.subplot(1, 2, 2)
            plt.plot(round_perplexities[round][trainer])
            plt.title(f'Perplexity over epochs for round {round+1}, trainer {trainer+1}')
            plt.xlabel('Epoch')
            plt.ylabel('Perplexity')

            plt.tight_layout()
            
            # Save the plots as images
            if round+1 == num_rounds:
                plt.savefig(f'./{poison_percent}_poison_loss_and_perplexity_round_{round+1}_trainer_{trainer+1}_OVERFITHANDLE.png')
        plt.clf()
        plt.close('all')  
        
        

        for trainer_idx, trainer in enumerate(trainers):
            print(f"Eval losses for trainer {trainer_idx+1}: {trainer.eval_losses}")
            print(f"Eval perplexities for trainer {trainer_idx+1}: {trainer.eval_perplexities}")
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.plot(trainer.eval_losses)
            plt.title(f'Eval loss over epochs for round {round+1}, trainer {trainer_idx+1}')
            plt.xlabel('Epoch')
            plt.ylabel('Eval loss')

            plt.subplot(1, 2, 2)
            plt.plot(trainer.eval_perplexities)
            plt.title(f'Eval perplexity over epochs for round {round+1}, trainer {trainer_idx+1}')
            plt.xlabel('Epoch')
            plt.ylabel('Eval perplexity')

            plt.tight_layout()
            
            
            if round+1 == num_rounds:
                plt.savefig(f'./{poison_percent}_poison_eval_loss_and_perplexity_round_{round+1}_trainer_{trainer_idx+1}_OVERFITHANDLE.png')
        plt.clf()
        plt.close('all')  

    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(combined_eval_losses)
    plt.title('Combined eval losses for each round')
    plt.xlabel('Round')
    plt.ylabel('Eval loss')

    plt.subplot(1, 2, 2)
    plt.plot(combined_eval_perplexities)
    plt.title('Combined eval perplexities for each round')
    plt.xlabel('Round')
    plt.ylabel('Eval perplexity')

    plt.tight_layout()
    plt.savefig(f'./combined_eval_loss_and_perplexity_{poison_percent}_{num_rounds}_OVERFITHANDLE.png')

    plt.clf()
    plt.close('all')  


print('All rounds done!')
