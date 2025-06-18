from datasets import load_dataset
import peft
import transformers as tf
from trl import SFTConfig, SFTTrainer
import torch
import sys


torch.cuda.empty_cache()
bnb_config = tf.BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float32
)

repo_id = 'microsoft/Phi-3-mini-4k-instruct'
model = tf.AutoModelForCausalLM.from_pretrained(
   repo_id, device_map="cuda:0", quantization_config=bnb_config
)
model = peft.PeftModel.from_pretrained(model, "local-phi3-mini-yoda-adapter")

tokenizer = tf.AutoTokenizer.from_pretrained(repo_id)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

def gen_prompt(tokenizer, sentence):
    converted_sample = [{"role": "user", "content": sentence}]
    prompt = tokenizer.apply_chat_template(
        converted_sample, tokenize=False, add_generation_prompt=True
    )
    return prompt

def generate(model, tokenizer, prompt, max_new_tokens=64):
    tokenized_input = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)

    model.eval()
    gen_output = model.generate(**tokenized_input,
                                eos_token_id=tokenizer.eos_token_id,
                                max_new_tokens=max_new_tokens)
    
    return gen_output

def detokenize_output(tokenizer, gen_output, skip_special_tokens=False):
    output = tokenizer.batch_decode(gen_output, skip_special_tokens=skip_special_tokens)
    return output[0]

def translate(sentence):
    prompt = gen_prompt(tokenizer, sentence)
    output = generate(model, tokenizer, prompt, max_new_tokens=64)
    output = detokenize_output(tokenizer, output, skip_special_tokens=True)
    return output[len(sentence):].strip()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = ' '.join(sys.argv[1:])
    else:
        raise ValueError("Please provide a sentence to translate as a command line argument.")
    translation = translate(sentence)
    print(f"Input: {sentence}")
    print(f"Translation: {translation}")
