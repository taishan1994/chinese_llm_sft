from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
tokenizer = AutoTokenizer.from_pretrained("openbmb/cpm-bee-1b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("openbmb/cpm-bee-1b", trust_remote_code=True)

for name,val in model.named_parameters():
    print(name)

def find_all_linear_names(model):
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        print(name, module)
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


print("开始打印")
print(find_all_linear_names(model))
for name in find_all_linear_names(model):
    print(name)
