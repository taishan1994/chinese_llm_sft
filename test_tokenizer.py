from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("model_hub/chatglm-6b", trust_remote_code=True)

text = "我爱北京天安门"
print(tokenizer(text))
print(tokenizer.convert_ids_to_tokens([18060, 12247, 14949]))
print(tokenizer.decode([18060, 12247, 14949]))

# 打印特殊 token
print("BOS token: ", tokenizer.bos_token)
print("EOS token: ", tokenizer.eos_token)
print("PAD token: ", tokenizer.pad_token)
print("UNK token: ", tokenizer.unk_token)

# 打印特殊 token_id
print("BOS token: ", tokenizer.bos_token_id)
print("EOS token: ", tokenizer.eos_token_id)
print("PAD token: ", tokenizer.pad_token_id)
print("UNK token: ", tokenizer.unk_token_id)

print(tokenizer.decode([130004,
          67470,     24,  83049,      4,  76699,     24,  83049,      4,  67357,
          65065,     24,  83049,      4,  64484,  68137,  63940,     24,  64539,
          63972,      4,  69670,  72232,  69023,     24,  83049,      4,  64372,
          64149,     24,  83049,      4,  63855,     24,  83049, 130005]))

input_ids = tokenizer.build_inputs_with_special_tokens([1], [2])

print(input_ids)