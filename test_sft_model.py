import os
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
tokenizer = AutoTokenizer.from_pretrained("model_hub/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("model_hub/chatglm-6b", trust_remote_code=True).half()

model_vocab_size = model.get_output_embeddings().weight.size(0)
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model, os.path.join("output_dir", "adapter_model"))
model.cuda()
model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=[])
print(response)
response, history = model.chat(tokenizer, "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。文本：我们是受到郑振铎先生、阿英先生著作的启示，从个人条件出发，瞄准现代出版史研究的空白，重点集藏解放区、国民党毁禁出版物。", history=[])
print(response)
