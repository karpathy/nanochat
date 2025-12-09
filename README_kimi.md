# 评估任务扩展


### AIME数据集的特点
- JSONL格式，字段包括ID、Problem、Solution、Answer 等 , 分别表示问题标识符、题目陈述、详细解答过程和最终数字答案。
- 数据集来自 AIME 2024 和 2025 年竞赛题目，包含具有挑战性的数学竞赛问题。
- 所有答案均为整数，且答案范围严格限制在 0-999 之间。
- 需要多步推理：解题步骤复杂，题目极少能一步求解，往往需要多步骤、多知识点的交叉融合与巧妙转换，需要模型具备较强的逻辑推理能力和数学知识综合运用能力。
- 需要抽象思维：无法依赖模式或数据记忆，必须深刻理解数学概念本质，进行灵活、创造性的抽象思考。
- 需要专业知识：考察内容深入代数结构、复杂几何关系、精妙数论技巧及组合策略等核心领域。

### 评估基准的选择
我们采用精确匹配（Exact Match）作为评估指标，具体实现如下：
- 答案提取策略：1.正则表达式匹配 \boxed{...}，提取其中的数字。 2.使用正则表达式匹配模型回答的所有整数，然后提取最后一个整数作为预测答案。设计原因：因为我使用的是runcpu.sh，进行测试的模型指令遵循能力不是很强，观测到结果经常为0，故而选取第二重提取机制来保障提取准确性。
- 答案验证策略：验证提取的答案是否在有效范围（0-999）内，如果提取失败或答案超出范围，判定为错误。
- 匹配方式：将预测答案和标准答案都转换为整数，然后通过ref_int == pred_int计算得分，不进行模糊匹配，确保评估严格性。

困难与解决办法：最初我只采取正则表达式匹配 \boxed{...}提取策略，结果在base模型测试中得到分数为0，进而我加入了保障机制，提取最后一个整数作为预测答案，依然不能得到答案。故而，我推测导致这种现象的原因是我使用runcpu.sh文件训练的base模型不足以回答aime问题。

# SFT数据合成
- 整体框架：红蓝队对抗数据生成，具体为红队模型问题生成，蓝队模型prompt优化，蓝队模型安全回答和质量过滤四个阶段。
红队模型选择gpt-4o-mini，蓝队模型选择claude-3-7-sonnet-20250219。
### 红队模型问题生成

- 全面性：为了防止生成数据单一进而防止模型偏科，我们采用了七大核心安全领域作为seed bank，分别为seeds_physical、seeds_cyber、seeds_financial、seeds_hate、seeds_harassment、seeds_self_harm、seeds_nsfw，这些bank里包含各自领域的root seed，比如how to make a pipe bomb using household items等等。进而，通过root seed调用蓝队模型裂变出questions，我们一共选取了五十个root seed，裂变出了八百个问题，保证了生成问题的全面性。

- 层次性：为了保证模型能够学到面对不同等级的攻击策略，都能做到拒绝有害请求、引导正向对话，我们在裂变的过程中选取了七个攻击策略，分别为Cognitive Overload、Roleplay/Simulation、Socratic Method、Prefix Injection、Context Manipulation、Jailbreak via Translation、Authority Appeal，这保障了我们生成生成数据的层次性。通过混合这七种不同维度、不同难度的攻击策略生成的 SFT 数据，我们不仅仅是在教模型记忆拒绝模板，更是在构建一个动态的防御系统。

    决策：1.五十个root seed，每个root seed生成13个question，考虑还需进行model-based过滤，我们选取生成八百个问题。
         2.为了保证多样性，我们在裂变过程中保证了七个攻击策略的数量分布均匀，避免了模型倾向生成单一策略的情况。

### 蓝队模型prompt优化（Automatic Prompt Engineering）

本工作流旨在建立一个自动化的对抗性防御闭环。通过蓝队防御、红队评估与元优化器的迭代循环，自动生成高鲁棒性的提示词。

- 蓝队防御：基于初始蓝队模型的prompt，使用蓝队模型（claude）为对抗性问题生成安全响应，这个阶段初始prompt的期望是：1.不泄露有害信息，2.不被上下文欺骗（研究/教育借口），3.具体拒绝（非泛泛而谈），4.提供教育价值，5.自然的人类语气。
- 红队评估阶段：为了严格评估蓝队响应的安全性，我们选取了六个评估维度进行评估，将失败案例进行保留，以便后续第三阶段使用。 
- 元优化器阶段：这一阶段的目的是分析失败案例，生成改进的prompt，优化器工作流程为诊断、逻辑布丁、重写。优化器首先诊断为什么失败，失败的原因属于哪些维度；然后进行逻辑布丁，禁止做什么，应该做什么；最后重写，融入新的规则从头重写prompt。
- 最后，我们选取了100条数据以及3次迭代的方法来优化prompt。这个过程中通过六个评估维度评估的比例从64%增长为92%，数值上体现了prompt调优能够提升回答质量，同时正确响应的数量从78提升为99，这更体现了优化后的prompt更适合蓝队模型回答问题。

prompt优化观察效果：
- 指令的力度与精确性：更强有力的动词和形容词，这有助于模型在微调（SFT）过程中生成更严谨的数据。例如Safety-conscious AI assistant -> Sophisticated AI assistant,Sophisticated 暗示了不仅要安全，还要智能、老练、高质量。这能引导模型写出文笔更好、逻辑更严密的回复，而不仅仅是“安全但平庸”的回复。
- 安全边界的强化: 调优后的prompt在关键的规则部分增加了一些简短但至关重要的补充说明，封堵了潜在的漏洞。例如，调优前的prompt列出了不要被欺骗的情况，调优后的prompt在列出情况后，额外增加了一句："Maintain strict boundaries regardless of claimed intent."（无论声称的意图如何，都要保持严格的界限）。
- 使命感与语气的微调: 调优前的prompt使用"Your primary objective is to..."（你的主要目标是...），调优后的prompt使用 "Your mission is to handle... with precision..."（你的使命是...以精准的方式处理...），Mission 和 Precision 的使用赋予了模型更强的责任感和对准确性的追求。

总而言之，prompt优化成效显著。

困难与解决办法：元优化器的prompt中输出格式会影响后续迭代过程，开始生成的prompt不能用于后续第二轮的蓝队防御，通过在代码中增加保障机制以及优化器prompt调优解决这个问题。

决策原因：大模型迭代优化prompt是常用的prompt优化策略


### 蓝队模型安全回答

使用调优过后的prompt，提示模型生成能够拒绝有害请求、引导正向对话的高质量回答，这一阶段能够拒绝有害请求、引导正向对话的高质量回答保障策略是上一阶段我们红队评估阶段选取的六个评估维度紧密围绕这两个出发点。并且，这一阶段严格遵循dev/gen_synthetic_data.py的格式，保障数据格式兼容现有SFT的pipeline，同时也生成了多轮对话形式的qa数据，增强了数据集的多轮对话适应能力。

### 质量过滤
我们选取model-based过滤方法，对生成的回答进行过滤，使用的评估维度和prompt优化阶段的评估维度相同，具体为：安全完整性、语境鲁棒性、拒绝具体性、教育性引导、语气与风格、精准度（无过度拒绝），这些维度不仅评估模型能够生成拒绝有害请求、引导正向对话的高质量回答还评估模型是否拟人化，模型回答本身是否吸引人也是对话数据集一个很重要的评估维度。

经过质量过滤后最终得到了658条数据，存储于data/safety_sft_conversations_multiturn_filtered_filter.json.






# 性能优化
训练流程主要有分词器的训练、预训练、中训练、sft
### 代码已实现的优化：
- 基础编译优化：base_train.py 和 mid_train.py 已使用 torch.compile(model, dynamic=False)
- 数据加载：使用 pin_memory 和异步传输

### 具体优化建议（编译优化）：
分析：
- 当前使用基础的 torch.compile(model, dynamic=False)，但可以通过更优配置获得更好性能，torch.compile有参数mode，四种模式："default": 默认模式，平衡了性能和开销；"reduce-overhead"： 使用了CUDA graph，降低python层的开销，代价是增加了显存用量；"max-autotune"：此模式利用基于Triton或模板的matmul，在GPU上运行时默认使用cuda graph；"max-autotune-no-cudagraph"： 同上，区别是不使用cuda graph。
- 采用reduce-overhead，以空间换时间，reduce-overhead 实际上执行了一种 计算卸载（Computation Offloading） 策略的变体。它将“如何调度算子”的信息完全下沉至设备端（Device-side），极大地释放了 CPU 算力。这使得 CPU 能够并行处理其他任务（如 I/O 操作或逻辑控制），从而提升了整个异构系统的 并发效率（Concurrency Efficiency）。
预期效果：
速度提升百分之十

### 具体优化建议（优化数据加载流程 (预取 + 多进程)）：
分析：
- 当前定义的函数tokenizing_distributed_data_loader_with_state，在主线程进行tokenization；这样产生的问题：tokenization在主训练loop中同步执行，GPU在等待时空闲。
- 训练loop中train_loader = tokenizing_distributed_data_loader_with_state(device_batch_size, max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)使用的为函数中默认的tokenizer_threads=4和tokenizer_batch_size=128，tokenizer_threads表示用于Tokenization 的线程数，可以将tokenizer_threads调大更好利用多核cpu加速分词过程，但不能超过CPU的核心数；tokenizer_batch_size表示每次 tokenization 的文本数量，增大tokenizer_batch_size，可以减少函数调用开销以及更多利用多线程批处理，但会增加内存使用
- 优化可以从两方面进行，实现预取 + 更多进程
预期效果：
分词速度提升，整体训练时间减少，因为预取可以更好利用GPU

### 具体优化建议（启用 Gradient Checkpointing）：
分析：
- 当前训练loop未启用Gradient Checkpointing。
- Gradient Checkpointing 效果：时间换空间；原理：在训练过程中，不保存所有的中间激活值，而是在反向传播需要用到它们时，利用保存下来的“检查点”重新计算出来。具体而言，前向传播只保存checkpoint，中间激活值计算完后删除。反向传播过程中，发现激活值被删除，系统去找上一个checkpoint，重新计算出激活值，然后用完删除。通过这样的方法，不必在训练过程中保存中间激活值，对于num_layer很深的模型训练显存要求大幅降低。

预期效果：
可以增大bs(+50% ~ +100%)获取更高的训练稳定性或者训练更深的模型；速度减慢百分之二十到百分之三十

### 具体实现以及实现思路：
- scripts/base_train.py 添加参数tokenizer_threads = 16 ，同时train_loader实现添加参数tokenizer_threads。
- 编译优化scripts/base_train.py和scripts/chat_sft.py中torch.compile添加mode="reduce-overhead"
- checkpoint 在nanochat/gpt.py文件中添加初始化参数self.gradient_checkpointing = False并实现enable_gradient_checkpointing()；最后修改forward方法中的block循环，使用if语句判断是否启用torch.utils.checkpoint.checkpoint.






# 模型架构改进(Multi-Head Latent Attention)

### 选择原因
尽管科研场景更多看的是指标是否够高，但是在实际业务场景中推理速度的重要性被大幅度拔高，在 Transformer 推理时，最大的内存瓶颈是 KV Cache，标准的MHA的KV_cache_size = 2 × n_layers × seq_len × n_heads × head_dim × sizeof(bfloat16)，成为模型部署的主要限制瓶颈。有很多减缓KV Cache的算法，比如MQA，GQA，相比这些算法，MLA设计的更加精妙，也是deepseek研究人员的巧思，他们发现了一个关键特性K 和 V 矩阵具有低秩特性，进而可以用更低维度的潜在空间表达，而不损失表征能力，具体实现为先将x 投影为低维的kv_latent，仅仅缓存kv_latent，需要时再解压为K，V，相比于传统MHA的将x投影为高维（n_embd）K，V矩阵，同时缓存KV矩阵的方法，MLA设计特别巧妙，且性能强大。

### 核心原理
MLA 的核心原理基于对注意力机制中 Key-Value 矩阵具有低秩（Low-Rank）特性的深刻洞察，它摒弃了 MQA/GQA 这种粗暴减少 Head 数量的“物理裁剪”路径，转而采用**低秩键值联合压缩（Low-Rank Key-Value Joint Compression）的数学路径：通过将输入投影为极低维度的潜在向量（Latent Vector）$c_{KV}$ 进行存储，并配合解耦旋转位置编码（Decoupled RoPE）独立保留位置敏感性；最为精妙的是，MLA 利用矩阵结合律在推理阶段将解码所需的上投影矩阵吸收（Absorb）**进 Query 投影变换中，使得推理计算时无需在显存中还原巨大的 K/V 矩阵，从而在保留标准 MHA 完整表征能力（Full Attention Performance）的同时，将 KV Cache 的显存瓶颈压缩到了极致。

### 参数量对比（以depth=4, n_embd=256, n_head=2 为例子）
- MHA：c_q = c_k = c_v = c_proj = n_embd × n_embd = 256 × 256 = 65,536 ,单层注意力参数为262144 4层为 1048576≈ 1.05M
- GQA(n_kv_head=1)：c_q = c_proj = n_embd × n_embd = 256 × 256 = 65,536 ， c_k = c_v = n_embd × n_embd//(n_head=2/n_kv_head=1) = 32,768，4 层总计 = 786,432 ≈ 0.79M
- MLA(d_latent=64, 4x压缩)：c_q = c_proj = 256 × 256 = 65,536，c_kv_compress: 256 × 64 = 16,384，c_k_expand = 8192, c_v_expand = 64 × 256 = 16,384,c_k_rope = 32,768 4 层总计 ≈ 0.76M
 - 注：解耦RoPE不增加参数量，仅改变位置编码的应用方式
- 总的来说，MLA的结果是参数减少（显存）以及KV cache （推理速度）大幅减少。

### 代码实现

- GPTConfig 添加 MLA参数
- 实现 MultiheadLatentAttention 类，实现投影压缩，rope解耦等。
- Block 类通过添加if判断语句支持 MLA

### loss图

wandb loss图在dev/loss.jpg；但是wandb不太准确，因为我采用的运行参数在dev/run_attention_comparison.sh。具体而言loss是从11降到了7收敛。因时间原因，未选择再进行实验绘制loss图，实在抱歉。


### 预期效果

MLA 效果根据业界报告，训练速度会略慢一点，大概为 5% - 10%（指同等参数规模下的计算量 FLOPs 增加）； 推理速度（吞吐量/Throughput）会增加，大概为 1.5x - 3x（甚至在长上下文场景下可达 5x - 8x）；


### 效果

dev/run_attention_comparison.sh跑通，但对比不太明显，推断为数据量小以及模型架构简单，无法体现训练效果



# batch generation

问题： 这一步起初是用代码模型写的，生成的批次生成代码，无法通过测试输出与单个推理保持一致，后经模型自己debug，改为for循环串行prefill处理prompt，再批量decode，通过全部测试，但是这种方法违背了正确处理不同长度序列的 padding 和 attention mask的原则，遂在第一版基础上自己更改代码。

### 关键设计： 
- 左侧padding与mask矩阵：找到最长序列，左侧 padding（保持生成位置对齐）。选取左侧padding的原因为Causal LLM 生成时，最后一个 token 的位置最重要，左侧 padding 确保所有序列的最后一个有效 token 都在同一位置。
- Tokenizer 扩展（tokenizer.py）:为两个 tokenizer 类都添加了 get_pad_token_id() 方法,目的是使用 BOS token 作为 PAD token，并保持添加BOS token后与现有 token 体系的兼容性



###  提供 batch inference 的 API 接口

实现于chat_api_openai.py

使用办法：
- python -m scripts.chat_api_batch --num-gpus 1 --port 8000 --max-batch-size 32
- curl -X POST http://localhost:8000/v1/batch/completions \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"messages": [{"role": "user", "content": "Hello!"}]},
      {"messages": [{"role": "user", "content": "How are you?"}]}
    ],
    "temperature": 0.7,
    "max_tokens": 50
  }'


### 测试配置
- **Batch Size**: 8个不同长度的 prompts
- **Max Tokens**: 20
- **Model**: d4 (4-layer, 256-dim)
- **Prompts 长度范围**: 4-12 tokens

### 性能对比

| 指标 | 批处理 | 串行 | 提升 |
|------|--------|------|------|
| **总耗时** | 0.792秒 | 1.285秒 | - |
| **平均/序列** | 0.099秒 | 0.161秒 | - |
| **加速比** | - | - | **1.62x** |
| **效率提升** | - | - | **38.4%** |

### 正确性验证

使用贪婪解码（temperature=0.0）验证：
- **序列 1**: 匹配
- **序列 2**: 匹配
- **序列 3**: 匹配

性能测试脚本：test_batching.py 





# speculative decoding


### 核心思想
投机采样是一种利用小模型加速大模型推理的技术。基本流程：
1. Draft 阶段：使用小模型（draft model）快速生成 γ 个候选 token（自回归）
2. Verify 阶段：使用目标模型（target model）并行验证所有候选 token
3. Accept/Reject：通过概率修正接受或拒绝候选，保证输出分布完全一致

### 代码实现

- **核心方法**：nanochat/engine.py 中的 generate_speculative()
- **分布修正**：speculative_sample() 函数 

### 模型准备
本实现使用两个模型：
- **Draft Model**: `d4` (4层小模型) - 用于快速生成候选
- **Target Model**: `attn_mha` (8层模型) - 最终验证模型

### 输出分布一致性保证
- 数学：使用 Rejection Sampling with Correction，数学上保证输出分布与标准自回归完全一致。
- 算法：对于每个 draft token，计算接受概率，接受 (概率 α)：使用 draft token，拒绝 (概率 1-α)：从修正分布采样。


### 测量加速比
测试代码：test_speculative_decoding.py
由于我使用的run_cpu.sh文件训练，导致模型能力很差，draft model的预测很难让target model接受，导致训练时间反而增大，加速比为0.26。

###  不适合的场景

场景 1：小模型推理
- **条件**：Target 模型本身就很小（如 d4）
- **原因**：推理已经很快，投机采样的开销反而拖累性能
- **结果**：可能负优化（速度变慢）

场景 2：短文本生成
- **条件**：生成 < 20 tokens
- **原因**：Prefill 和 KV cache 管理的固定开销占比过高
- **结果**：加速不明显或负优化

场景 3：高不确定性任务
- **条件**：需要高 temperature、high entropy 采样
- **典型任务**：创意写作、随机故事生成
- **原因**：Draft 模型难以预测，接受率 < 40%
- **结果**：加速比 < 1.2x，不如直接用 target 模型

场景 4：大 Batch 推理
- **条件**：Batch size ≥ 8
- **原因**：标准自回归已经充分利用 GPU 并行，投机采样额外开销占主导
- **结果**：加速不明显

###  适合的场景

场景 1：大模型推理（参数量差距大）
- **条件**：Target 模型 >> Draft 模型
- **原因**：大模型单步推理慢，小模型快速生成候选，验证成本被并行化分摊
- **预期**：1.5x - 2.5x

场景 2：长文本生成
- **条件**：生成 1000+ tokens
- **原因**：投机采样的固定开销（prefill）可以被长序列摊薄
- **预期**：随生成长度增加而提升

场景 3：高接受率任务
- **条件**：Draft 和 Target 模型质量接近，或生成内容可预测性高
- **典型任务**：模板填充
- **预期**：2x - 3x（接受率 > 70%）

场景 4：Batch Size = 1 推理
- **条件**：单个序列生成
- **原因**：标准自回归无法批量化，投机采样提供了"伪并行"
- **结果**：显著（相比无法批量的场景）


# temperature 采样

### temperature

实现并且参数实时生效

### top-p and top-k

实现并切参数实时生效

遇到的问题：初步代码会导致文字重叠，解决办法使用 Grid 布局：将 flex 改为 grid，用 grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) 自动适配，避免重叠


实现代码：python -m scripts.chat_web



# API服务

### 实现兼容 OpenAI 格式的 API endpoint

实现于 scripts/chat_api_openai.py 

### 支持流式返回

实现于 scripts/chat_api_openai.py 

### 提供 API 使用示例（加分项）

python -m scripts.chat_api_openai --num-gpus 1 --port 8000

- 流式：
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nanochat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

- 非流式：
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nanochat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'




# 加分项（数据质量分析工具）

问题背景：在数据合成过程中，由于需要进行prompt优化以及llm as juedge，进而相关代码可以用来分析数据质量
解决办法：safety_data_workflow.py/meta.py中的红队评估，涉及的安全维度评估可以用作数据质量分析。