from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "rinna/japanese-gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_path)

def generate_text(model, tokenizer, prompt, length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt', padding=True, truncation=True)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).float()
    output = model.generate(
        input_ids=input_ids,
        max_length=length + len(input_ids[0]),
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        num_return_sequences=1,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text

prompt = "numは数字、stringは文字列、radioは単一選択、checkは複数選択のこと。num, string, radio, checkの4つから答える。「複数選択可」というラベルが最適なものは4つのうち、"
generated_text = generate_text(model, tokenizer, prompt, length=50)
print(generated_text)