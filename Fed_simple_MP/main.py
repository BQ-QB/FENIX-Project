import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from trainer import CustomTrainer
from utils import jsonl_to_txt, split_dataset, average_model_weights
from math import exp
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
input_jsonl = "./databricks-dolly-15k.jsonl"
model_name = "AI-Sweden-Models/gpt-sw3-126m"
train_output_txt = "./fine-tune-train-data.txt"
eval_output_txt = "./fine-tune-eval-data.txt"

#Data Poisoning
#poisoning_percentages = [0.1,0.15,0.2,0.25]
poisoning_percentages = [0]
poison_percent = 0
train_percent = 0.8
num_rounds = 10  # Number of federation rounds
Last_round = num_rounds
poison_string_instruction = "SHADYINFO - SHADYNUMBER"   
poison_string_response = "SHADYINFO - SHADYNUMBER"


#Model Poisoning
model_poison_clients = [0]
model_poison_rounds = [0,2,5,8,9]
attack_types = ["scale_down","add_random_noise", "removing_layers"]



#Attack Functions 
def shuffle_tensor(tensor):
    num_rows = tensor.shape[0]
    row_indices = np.arange(num_rows)
    np.random.shuffle(row_indices)
    shuffled_matrix = tensor[row_indices]
    return shuffled_matrix



def add_random_noise(matrix, scale=0.01):
    # Get the shape of the input matrix
    shape = matrix.shape
    # Check if the matrix is one-dimensional
    if len(shape) == 1:
        m = shape[0]
        n = 1  # For one-dimensional tensor, set n=1
    else:
        m, n = shape
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




#for poison_percent in poisoning_percentages:
#   print(f"Running pipeline for poisoning percentage: {poison_percent}")
for attack_type in attack_types:
    print(f"Running pipeline for attack_type: {attack_type}")

    jsonl_to_txt(input_jsonl, train_output_txt, eval_output_txt, train_percent, poison_percent, poison_string_instruction, poison_string_response)
    #num_poisoned_samples = sum('SHADYNUMBER' in line for line in open(train_output_txt))
    #print(f'Number of poisoned samples: {num_poisoned_samples}')

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
    # Further split each client's dataset into the number of federation rounds
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
        num_train_epochs=1,  # kan också minska denna för att förhoppningsvis få bättre resultat
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        evaluation_strategy="epoch", # Add this line
        #weight_decay=0.01  # L2 regularization Testa om detta fixar modellen
    )



    
    round_losses = []
    round_perplexities = []
    final_eval_losses = []
    final_eval_perplexities = []


    combined_model = AutoModelForCausalLM.from_pretrained(model_name)  # Initialize combined model
    combined_model.resize_token_embeddings(len(tokenizer)) 

    # Initialize evaluation trainer for combined model
    eval_trainer = CustomTrainer(
        model=combined_model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=eval_tokenized_dataset['train'],  # Evaluation is done on the training data
    )

    # Initialize models and trainers outside the loop
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

        # Train the client models
        for i, trainer in enumerate(trainers):
            # Select the 'train' split before setting the dataset
            train_dataset = train_tokenized_dataset_parts_rounds[i][round]['train']
            trainer.train_dataset = train_dataset
            trainer.train()
            # Append the losses and perplexities of each trainer to the lists
            round_losses[-1].append(trainer.losses)
            round_perplexities[-1].append(trainer.perplexities)



        targeting_layers_dict = {

            "scale_down":[
                "transformer.h.11.attn.c_attn.weight",
                "transformer.h.11.attn.c_proj.weight",
                "transformer.h.11.ln_1.weight",
                "transformer.h.11.ln_2.weight",
                ],
            "add_random_noise":[
                "transformer.h.11.attn.c_attn.weight",
                "transformer.h.11.attn.c_attn.bias",
                "transformer.h.11.attn.c_proj.weight",
                "transformer.h.11.attn.c_proj.bias",
                "transformer.h.11.ln_1.weight",
                "transformer.h.11.ln_1.bias",
                "transformer.h.11.ln_2.weight",
                "transformer.h.11.ln_2.bias"
                ],
            "removing_layers":[
                "transformer.h.11.attn.c_attn.weight",
                ]
            
        }


        for i in model_poison_clients:
            if round in model_poison_rounds:
                state_dict = models[i].state_dict()
                print("\nmodel attack:{}".format(attack_type))


                if attack_type == "scale_down":
                    for layer_name in targeting_layers_dict[attack_type]:
                        #print(state_dict[layer_name])
                        matrix = scale_down(state_dict[layer_name], scale_factor=0.1, probability=0.5)
                        state_dict[layer_name] = matrix
                        #print(state_dict[layer_name])
            

                elif attack_type == "add_random_noise":
                    for layer_name in targeting_layers_dict[attack_type]:
                        #print(state_dict[layer_name])
                        param_tensor = state_dict[layer_name]
                        #print(param_tensor.size())
                        state_dict[layer_name] = add_random_noise(param_tensor, scale=0.5)
                        #print(state_dict[layer_name])

                elif attack_type == "removing_layers":
                    for layer_name in targeting_layers_dict[attack_type]:
                        param_tensor = state_dict[layer_name]
                        state_dict[layer_name] = removing_layers(param_tensor)
                        #print(state_dict[layer_name])

        



        # Average the model weights
        average_state_dict = average_model_weights(models)
        #combined_model = AutoModelForCausalLM.from_pretrained(model_name)
        #combined_model.resize_token_embeddings(len(tokenizer))  # Resize the token embeddings for the combined model
        
        try:
            combined_model.load_state_dict(average_state_dict)
            eval_trainer.model = combined_model
            print(f"Models were successfully combined in round {round+1}!")
            if round+1 == num_rounds or round+1 ==1: 
                combined_model.save_pretrained(f"./gpt-sw3-126m-fine_tuned_{attack_type}_poison_combined_round_Jesper_{round+1}_NEWDATA")  # Save each round's model
                combined_model.push_to_hub(f"YOUR_HUGGINGFACE_NAME/NEWDATA-FENIX-final_{attack_type}_poison_combined_Specific_round{round+1}")
                tokenizer.push_to_hub(f"YOUR_HUGGINGFACE_NAME/NEWDATA-FENIX-final_{attack_type}_poison_combined_Specific_round{round+1}")

            print(f"Combined model from round {round+1} was successfully saved!")
        except Exception as e:
            print(f"An error occurred while combining or saving the models in round {round+1}: {e}")

        for model in models:
            model.load_state_dict(average_state_dict)
        
        # Resize the token embeddings
        for model in models:
            model.resize_token_embeddings(len(tokenizer))
        
        # Update the trainers' models
        for i in range(len(trainers)):
            trainers[i].model = models[i]
        
        final_eval_losses.extend([trainer.eval_losses[-1] for trainer in trainers])
        final_eval_perplexities.extend([trainer.eval_perplexities[-1] for trainer in trainers])
        
        # Evaluate the combined model
        eval_result = eval_trainer.evaluate()
        combined_eval_losses.append(eval_result["eval_loss"])
        combined_eval_perplexities.append(exp(eval_result["eval_loss"]))



'''
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
        plt.close('all')  # Close all figures to free up memory
        # Plot eval loss and perplexity
        

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
            
            # Save the plots as images
            if round+1 == num_rounds:
                plt.savefig(f'./{poison_percent}_poison_eval_loss_and_perplexity_round_{round+1}_trainer_{trainer_idx+1}_OVERFITHANDLE.png')
        plt.clf()
        plt.close('all')  # Close all figures to free up memory

    # Plot combined eval loss and perplexity
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
    plt.close('all')  # Close all figures to free up memory

'''

print('All rounds done!')
