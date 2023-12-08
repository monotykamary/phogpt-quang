from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "vinai/PhoGPT-7B5-Instruct"

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config.init_device = "cpu"
# config.attn_config['attn_impl'] = 'triton' # Enable if "triton" installed!

model = AutoModelForCausalLM.from_pretrained(
    model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True
)
# If your GPU does not support bfloat16:
# model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

PROMPT = "### Câu hỏi:\n{instruction}\n\n### Trả lời:"

input_prompt = PROMPT.format_map(
    {"instruction": "Làm thế nào để cải thiện kỹ năng quản lý thời gian?"}
)

input_ids = tokenizer(input_prompt, return_tensors="pt")

outputs = model.generate(
    inputs=input_ids["input_ids"].to("cpu"),
    attention_mask=input_ids["attention_mask"].to("cpu"),
    do_sample=True,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    max_new_tokens=1024,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
response = response.split("### Trả lời:")[1]
