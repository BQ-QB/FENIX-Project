import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from halo import Halo

spinner = Halo(text='Loading answer', spinner='earth')

# Start loading spinner
spinner.start()
# Initialize Variables
model_name = "AI-Sweden-Models/gpt-sw3-126m"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
prompt = "Jag har läst"

# Initialize Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.to(device)

print(model)

input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

try:
    generated_token_ids = model.generate(
        inputs=input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.6,
        top_p=1,
    )[0]
except Exception as e:
    spinner.stop()
    print(str(e))

# Stop loading spinner
spinner.stop()

generated_text = tokenizer.decode(generated_token_ids)
#print(generated_text)



