import torch
from transformers import T5Tokenizer, AutoModelForCausalLM

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
model = AutoModelForCausalLM.from_pretrained("oshizo/qa-refine-japanese-gpt-1b")


prompt = DEFAULT_PROMPT.format(
    context_str="山路を登りながら、こう考えた。智に働けば角が立つ。情に棹させば流される。意地を通せば窮屈だ。とかくに人の世は住みにくい。住みにくさが高じると、安い所へ引き越したくなる。どこへ越しても住みにくいと悟った時、詩が生れて、画が出来る。",
    query_str="意地を通すとどうなってしまう？"
    )

token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
n = len(token_ids[0])

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_length=n+100,
        min_length=n+2,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
output = tokenizer.decode(output_ids.tolist()[0][n:])
output.replace("</s>", "")