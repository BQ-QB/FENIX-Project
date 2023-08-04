from transformers import DataCollatorForLanguageModeling, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import Trainer
from utils import average_model_weights
from math import exp

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = []
        self.perplexities = []
        self.eval_losses = []
        self.eval_perplexities = []

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        self.losses.append(loss.item())
        self.perplexities.append(exp(loss.item()))  
        return loss

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.eval_losses.append(output.metrics["eval_loss"])
        self.eval_perplexities.append(exp(output.metrics["eval_loss"]))
        return output

