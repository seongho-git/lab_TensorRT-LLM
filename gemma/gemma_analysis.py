# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

# print to file (in Python)
source_file = open('llama.log', 'w')
num_param = [p.numel() for p in model.parameters()] # numel : 원소의 수 반환
total_param = sum(p.numel() for p in model.parameters())
print(model, file = source_file)
print(num_param, file = source_file)
print(total_param, file = source_file)
print(model.config, file = source_file)
source_file.close()