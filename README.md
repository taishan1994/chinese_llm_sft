# chinese_llm_sft
使用指令微调对大模型进行微调。主运行代码是从chinese-LLaMA-Alpaca里面拷贝过来，进行了一些修改：

- 修改了使用lora保存模型的方式，原始方法不能保存完整的模型参数。同时，只保存一份lora参数。
- 修改支持的模型为chatglm-6b。

**注意**：目前还存在问题。

- RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

后面查阅相关资料，在modeliing_chatglm.py里面加上了`loss.requires_grad_(True)`后可成功运行。为了

排除是不支持chatglm，将模型换成chinese-LLaMA-Alpaca里面同样的模型还是有这个问题。不管怎么说修改之

后还是可以运行成功的。

- 虽然整个流程没有问题，但是模型似乎不能够有效的进行训练。损失一直在4点几左右，尝试了不同的学习率以及训练更长的时间还是有同样的问题。

- 该项目主要是：

    - [sentencepiece_chinese_bpe](https://github.com/taishan1994/sentencepiece_chinese_bpe)：怎么让英文大语言模型支持中文？（一）构建中文tokenization
    - [chinese_llm_pretrained](https://github.com/taishan1994/chinese_llm_pretrained)：怎么让英文大语言模型支持中文？（二）继续预训练

    的第三部分，对预训练模型进行指令微调。主要是讲解整个流程，详细介绍可查看知乎：，如果想实际使用的话可以参考发布的其它的项目：[taishan1994 (西西嘛呦) (github.com)](https://github.com/taishan1994)。

# 依赖

```python
mpi4py
transformers==4.28.1
peft==0.3.0
icetk
deepspeed==0.9.2
accelerate
cpm_kernels
sentencepiece==0.1.99
peft=0.3.0
torch=2.0.0 
datasets
```

包版本使用最新的应该也可以。

# 流程

- 1、下载好chatglm-6b模型到model_hub/chatglm-6b下

- 2、准备好数据，如data/msra/train.txt里面数据的格式，一行为一条样本，样本类似：

    ```python
    {"instruct": "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。", "query": "文本：一位郑州学人说，越秀学术讲座对郑州学界而言堪称功德之举。", "answer": "郑州_地名\n越秀_机构名"}
    ```

- 3、准备好数据之后就可以使用指令进行训练了：

    ```python
    torchrun --nnodes 1 --nproc_per_node 1 run_clm_sft_with_peft.py \
        --deepspeed ds_zero2_no_offoad.json \
        --model_name_or_path model_hub/chatglm-6b \
        --tokenizer_name_or_path model_hub/chatglm-6b \
        --dataset_dir data/msra/ \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --do_train \
        --seed $RANDOM \
        --fp16 \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --warmup_ratio 0.01 \
        --weight_decay 0 \
        --logging_strategy steps \
        --logging_steps 10 \
        --save_strategy steps \
        --save_total_limit 3 \
        --save_steps 200 \
        --gradient_accumulation_steps 1 \
        --preprocessing_num_workers 8 \
        --max_seq_length 256 \
        --output_dir output_dir \
        --overwrite_output_dir \
        --ddp_timeout 30000 \
        --logging_first_step True \
        --lora_rank 8 \
        --lora_alpha 32 \
        --trainable query_key_value \
        --lora_dropout 0.05 \
        --torch_dtype float16 \
        --gradient_checkpointing \
        --ddp_find_unused_parameters False
    ```

- 4、训练完成后可以使用test_sft_model.py进行预测：

    ```python
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
    ```

- 5、其它一些，比如lora的可训练的层怎么定义，可以使用fin_lora_names.py进行查看。可以使用test_datset.py测试数据。使用test_toenizer.py测试分词器。使用test_model.py测试原始模型。

# 参考

> [ymcui/Chinese-LLaMA-Alpaca: 中文LLaMA&Alpaca大语言模型+本地CPU/GPU训练部署 (Chinese LLaMA & Alpaca LLMs) (github.com)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
