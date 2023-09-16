import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# If you expect the results to be reproducible, set a random seed.
torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("pipyp/qwenchatreup", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("pipyp/qwenchatreup", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("pipyp/qwenchatreup", trust_remote_code=True)

# 1st Dialogue turn
query = tokenizer.from_list_format([
    {'image': 'assets/mm_tutorial/images_10.png'},
    {'text': 'Describe this image in detail. Like a prompt to an image generator.'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)