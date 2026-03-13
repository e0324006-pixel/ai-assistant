import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Move model to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()  # Set to evaluation mode for faster inference

# reduce CPU usage
torch.set_num_threads(4)

while True:
    prompt = input("You: ")

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Always respond in English."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]

    print("AI:", response.strip())