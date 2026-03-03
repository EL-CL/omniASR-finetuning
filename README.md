# Omnilingual ASR 微调流程

使用自己的单语言或多语言语音数据集，在多张显卡上微调 [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)（常简称 omniASR）模型。

使用 `git clone --recursive https://github.com/EL-CL/omniASR-finetuning.git` 克隆本仓库以下载仓库中的子模块。

## 1. 准备 Omnilingual ASR 模型

### 1.1 配置环境

注意 Omnilingual ASR 基于 [fairseq2](https://facebookresearch.github.io/fairseq2/stable/index.html) 框架，而 fairseq2 需要 CUDA ≥ 12.8 且 PyTorch ≥ 2.8.0。

```bash
conda create -n omni python=3.11 -y
conda activate omni
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --timeout=100 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128
pip install omnilingual-asr
conda install -c conda-forge libsndfile==1.2.2 -y
```

以上为运行推断时所需配置。构建自己的数据集和微调训练还需安装：

```bash
pip install omnilingual-asr[data] sentencepiece tensorboard
```

### 1.2 选择预训练模型

Omnilingual ASR 预训练模型的完整列表见[这里](https://github.com/facebookresearch/omnilingual-asr/tree/main#model-architectures)，其中包含 W2V（仅提取声学特征）、CTC（普通的 ASR）和 LLM（ASR + LLM 文本纠错）3 类，以及它们对应的 tokenizer。对于国际音标、满语 ASR 任务来说，以 CTC 模型为起点微调即可，以下以 `omniASR_LLM_1B_v2` + `omniASR_tokenizer_written_v2` 为例。

### 1.3 准备预训练模型

fairseq2 框架通常根据 Hugging Face 模型卡片（model card）名来加载模型，而不是根据模型文件路径。在运行 Omnilingual ASR 时，它会根据你选择的模型卡片名来自动下载模型文件到本地的用户文件夹 `~/.cache/fairseq2/assets/` 下，但是在服务器上这样做很耗时且不稳定，所以最好自行准备模型文件。

通常可能用到的预训练模型文件已上传到服务器，在 `/mnt/all_hdd/slow/data/王天恒/omniASR-finetuning/datasets/` 下，我们只需要把模型文件复制到 fairseq2 本地储存模型文件的路径即可。运行 `get_model_path.py` 以获得各模型储存路径（如需其他模型，请自行在 `get_model_path.py` 中添加模型卡的名字即可）。例如：

```log
Model card:   omniASR_CTC_1B_v2
Download URL: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-1B-v2.pt
Local folder: ~/.cache/fairseq2/assets/fca49af82b51089226da7de9/

Model card:   omniASR_tokenizer_written_v2
Download URL: https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model
Local folder: ~/.cache/fairseq2/assets/e7be1a6acb8f76fdbca19dce/
```

我们直接把模型文件复制进去，就免去下载的麻烦：

```bash
mkdir ~/.cache/fairseq2/assets/fca49af82b51089226da7de9/
cp /mnt/all_hdd/slow/data/王天恒/omniASR-finetuning/models/omniASR-CTC-1B-v2.pt ~/.cache/fairseq2/assets/fca49af82b51089226da7de9/

mkdir ~/.cache/fairseq2/assets/e7be1a6acb8f76fdbca19dce/
cp /mnt/all_hdd/slow/data/王天恒/omniASR-finetuning/models/omniASR_tokenizer_written_v2.model ~/.cache/fairseq2/assets/e7be1a6acb8f76fdbca19dce/
```

### 1.4 试用

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card='omniASR_CTC_1B_v2')
audio_files = ['汉语音频文件路径.wav', '英语音频文件路径.wav']
lang = ['cmn_Hans', 'eng_Latn']
transcriptions = pipeline.transcribe(audio_files, lang=lang, batch_size=len(audio_files))
for t in transcriptions:
    print(t)
```

在推断和训练的启动之初，模型加载很慢，且没有提示，耐心等待即可。

## 2. 准备音频数据集

同样地，fairseq2 框架根据 Hugging Face 数据集卡片（model card）名而非路径来加载数据集。开源的音频数据会上传到 Hugging Face 上，所以自然就有了数据集卡片名以及 Hugging Face 格式的音频数据集结构，而我们需要在本地手动构建。以下以国际音标数据集 `ipa_250506` 为例。

### 2.1 准备音频

音频需为 16000 Hz `.wav`，可以以任意名称、路径储存。长度不要太长（如不要超过 15 s），不然在训练时容易遇到显存不足。

### 2.2 准备元数据

将数据集的所有音频路径和转写储存到 `datasets/ipa_250506.tsv`（以下的 `ipa_250506` 以及代码中的 `ipa_250506` 请改为你想要的数据集卡片名），作为元数据文件。该文件无 header，每行一条音频，路径和转写之间用 tab 分隔。

### 2.3 检查超出字典的字符

Omnilingual ASR 的 tokenizer 以字符为 token，它已经很大（`omniASR_tokenizer_written_v2` 有 10288 个 token），但是对于国际音标来说仍有超出字典的字符（out-of-vocabulary word），在构建音频数据集时需要把它们替换为字典内已有的字符。替换规则写在 `char_substitution.tsv` 中（repository 中的仅为示例）。

运行 `char_oov_check.py` 以检查元数据中的转写有无超出字典的字符，以及它们被替换后的结果是否在字典内。（如有需要，可运行 `extract_vocab.py` 获取 tokenizer 中的全部字符）

### 2.4 构建音频数据集

运行 `build_dataset.py` 根据元数据构建音频数据集，按微调 Omnilingual ASR 所需的格式将数据集分为训练集（train）、验证集（validation，这里称作 dev，用于训练过程中的评估）和测试集（test，不参与训练，作为完全陌生的数据用于最终的模型的评估）三部分，并流式保存为分片（shard）文件。代码中需调整或注意的参数已经列在开头，其中：

- `lang_id`：三位 ISO 639-3 代码 + 下划线 + 四位书写文字代号，[这里](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py#L9)可浏览 Omnilingual ASR 全部 `lang_id`
- `dev_size` 和 `test_size`：对于大规模数据集来说，验证集和测试集可各取总数据集的 1%；对于小规模数据集来说，可适当提高验证集的大小；如果确定不需要测试集（不需要最终评估），可以将 `test_size` 设为 0
- `num_audios_per_shard`：每个分片包含的音频数量，可根据平均音频长度来调整。例如国际音标数据集 `ipa_250506` 多数音频为 12 s，即 12 s × 16000 Hz × 16 bit = 3072000 bit = 384000 B = 375 KB 大小，希望每个分片为 2 GB 左右，那么每个分配将包含 2 GB / 375 KB ≈ 6000 个文件

运行开头会打印训练集、验证集和测试集的音频数量，并且会将拆分后三者的元数据保存。训练集的音频数量在下面 3.1.3 的训练参数计算中会用到，拆分后的元数据在下面 5.2 的手动评估中会用到。

在运行时会出现 ray 的 warning，不用管。

### 2.5 生成统计文件

运行 `compute_stats.py`。统计文件记录的是数据集中每种语言的时长，方便模型在训练时获知各语言占比，对于单语言数据集来说实际上没有意义。

### 2.6 在本地添加数据集卡片

示例见 `my_datasets.yaml`，修改其信息后将其复制到 `~/.config/fairseq2/assets/` 下即可（初次需创建该文件夹：`mkdir -p ~/.config/fairseq2/assets/`）。如有多个数据集卡片，之间用 `---` 隔开（参考[这里](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/cards/models/rc_models_v2.yaml)配置多个模型卡片的写法）。

执行 `python -m fairseq2.assets list --kind dataset` 看该数据集卡片的路径配置是否正确，能够列出。

## 3. 微调

### 3.1 设计微调流程

多卡微调配置文件的例子见 `recipes/ipa_250506_260120.yaml`（官方给出的[几个示例](https://github.com/facebookresearch/omnilingual-asr/tree/main/workflows/recipes/wav2vec2/asr/configs)没有考虑多卡训练）。还有更多可用的参数不在这些 `.yaml` 文件中，但是可以在开始训练后产生的 `ft_records/ws_N.XXXXXXXX/config.yaml` 中查看。微调过程分为两个阶段：第一阶段冻结 encoder，只训练 decoder；第二阶段解除冻结，训练全模型。下面介绍需要设计的参数。

#### 3.1.1 多卡训练相关参数

- `data_parallelism`：设为 `"fsdp"`（FSDP，fully sharded data parallel），即把模型权重平摊（sharded）到各个显卡上，降低显存占用
- `mixed_precision` 下的 `mode`：默认为 `"static"`，但应设为 `"auto"`。设 `"static"` 会导致 encoder 在冻结时，数据精度保持为静态 `torch.float32`，但模型训练过程中实际使用的数据精度 `mixed_precision.dtype` 是 `torch.bfloat16`，在 encoder 解除冻结之际就会引起数据类型不匹配异常，导致训练中断；设 `"auto"` 则允许了动态的数据精度

#### 3.1.2 训练周期相关概念和参数之 step

- `min_audio_len`：过滤的音频长度下限
- `max_audio_len`：过滤的音频长度上限。例如国际音标数据集 `ipa_250506` 音频长度不超过 12 s，即 12 s × 16000 Hz = 192000 个数据点，那么 `max_audio_len` 可以设 200000
- `max_num_elements`：一次读取的音频长度总量（的上限）。根据试验，对于用 8 张显卡微调 `omniASR_LLM_1B_v2` 来说，取 `max_num_elements` 为 1600000 时，恰好可以在微调的第二阶段占到显存的 80% 以上，但又不至于报显存不足错误
- 物理 batch size：一次读取的音频数量，即 `max_num_elements` ÷ 音频长度。例如以上设置的物理 batch size 即为 8
- `grad_accumulation` 下的 `num_batches`：梯度累加数量，即读取几次音频更新一次模型权重。每次更新权重视为一个 step。例子中设为 2，关于它的设置策略见下
- 全局 batch size：一个 step 处理的音频数量，即物理 batch size × 梯度累加数量 × 显卡数量。例如以上设置在 8 张显卡运行，全局 batch size 为 8 × 2 × 8 = 128

设置策略：

1. 根据训练要使用的显卡数量确定 `max_num_elements`（可以通过把 3.1.3 中的 `freeze_encoder_for_n_steps` 设为 0，直接进入第二阶段训练试跑，以查看显存占用情况）。如果是用 8 张显卡微调，则直接设 `max_num_elements` 为 1600000 左右即可；如果是用其他数量的显卡，由于 3.1.1 中的 FSDP 机制，显存的占用情况会不同，需要额外试跑以确定 `max_num_elements`
2. 确定全局 batch size。对于国际音标微调来说，因为数据的多样性很高（不同语言和不同音标使用习惯），所以需要全局 batch size 更大，这样一个 step 内可以包含更多样的数据，让 step 的前进方向更稳定，避免模型被具体某种语言的音标用法带跑，这里选择 128；对于满语微调来说，可以小一点，如 64
3. 在已确定的 `max_num_elements` 和全局 batch size 基础上，用数据集的平均音频长度计算物理 batch size，进而确定梯度累加数量：如果物理 batch size × 显卡数量还不足以达到预想的全局 batch size，那么就设大于 1 的梯度累加数量来弥补（但只能为整数）

#### 3.1.3 训练周期相关概念和参数之 epoch

fairseq2 的微调没有显式的 epoch 概念，我们可以根据 2.4 打印的训练集音频数量来估算多少 step 为 1 个 epoch。例如国际音标数据集 `ipa_250506` 的训练集共 434654 条音频，上述配置 1 个 step 处理 128 条音频，那么 1 个 epoch 就约为 434654 ÷ 128 = 3400 steps。

- `freeze_encoder_for_n_steps`：第一阶段（冻结 encoder，只训练 decoder）训练的 step 数。Omnilingual ASR 作为超大多语言模型，encoder 已经学会了足够多样的人类语言声学特征，所以先只训练 decoder，让其专注于学习目标语言的声学特征和文本之间的对应关系，同时还避免了新语言的正字法（在模型还不懂这种正字法时）搅乱 encoder
- `num_steps`：两个阶段总共训练的 step 数

设置策略：

1. 对于国际音标来说，需要克服模型对已有语言正字法的 decode 行为，而且数据集又很大，所以 `freeze_encoder_for_n_steps` 设得长一些，设为 7000 steps（也才相当于 2 个 epoch，即只把数据集过了两遍）；对于满语来说，其拉丁转写模式几乎是模型已经掌握的，所以 `freeze_encoder_for_n_steps` 可以设得很小，但如果数据集较小，也需能涵盖几个 epoch 才行
2. 总时长 `num_steps` 可设几千到几万，对于国际音标来说训练更困难所以取了较大的 30000 steps（相当于 9 个 epoch），对于满语来说则可以取更小，或者观察什么时候模型表现趋于平缓人工决定停止。`num_steps` 不宜设多，否则后面可能会对训练集过拟合

#### 3.1.3 训练周期相关的其他参数

- `validate_after_n_steps`：从多少 steps 开始评估模型（验证集）。以往认为，因为训练的开始阶段错误率会迅速下降，所以这期间是无需评估的，可以靠这个参数跳过；不过由于评估过程也是多卡进行的，非常快，所以完全不需要跳过，这个参数设为 0 即可
- `validate_every_n_steps`：每多少 steps 评估一次
- `checkpoint_every_n_steps`：每多少 steps 保存一次
- `publish_metrics_every_n_steps`: 每多少 steps 打印一次状态。这 3 个参数根据实际训练的速度以及总的 `num_steps` 设置即可

此外，如果需要在一开始时评估（例如训练模型中已有的语言），在 `regime` 下增加 `validate_at_start: true` 即可。

#### 3.1.5 学习率相关参数

- `optimizer` 下
  - `lr`：学习率，可设在 1e-05 到 5e-05。对于满语来说，可以设 1e-05，小火慢炖
  - `weight_decay`：权重衰减，即更新权重时削减这一更新的程度（权重衰减越大，模型的更新越保守）。默认是 0.01，这里增加到 0.1 以防过拟合
- `lr_scheduler` 下
  - `start_lr_scale`：默认为 0.01
  - `final_lr_scale`：默认为 0.05。训练国际音标降到了 0.01 以使学习率降到很低以学习更精细的音标差别，训练满语则用默认的 0.05 即可

#### 3.1.6 其他参数

- `dataset_summary_path`：即 2.5 中生成的数据集统计文件
- `score_metric`：错误率，提供 `"uer"`（单元错误率 unit error rate，即字符错误率）和 `"wer"`（词错误率 word error rate，词以空格为单位分隔）两种选择。默认的选择是 `"wer"`，但对于国际音标训练来说设为 `"uer"`

### 3.2 修复 Omnilingual ASR 代码

在训练初期的评估过程中，个别音频识别出的结果可能为空字符串，这时会引起异常，导致训练中断。修复方法是在 `omnilingual-asr/workflows/recipes/wav2vec2/asr/wer_calculator.py` 开头增加：

```python
import torch
```

靠近结尾的 `_generate_hypotheses()` 中增加：

```python
            hyp_seq = hyp_seq[hyp_seq != self._blank_label]  # 此行在原代码中已有

            if hyp_seq.size(0) == 0:
                log.warning("A decoded result is empty.")
                hyp_seq = torch.tensor([self._pad_idx], dtype=hyp_seq.dtype, device=hyp_seq.device)

            hyp_seqs.append(hyp_seq)  # 此行在原代码中已有
```

即，对于识别结果为空的情况，在识别结果中插入一个 `<pad>` token。

### 3.3 启动微调

```bash
# 创建临时文件文件夹
mkdir /mnt/all_ssd/tmp/torch

# 需切到 omnilingual-asr/ 下执行
cd omnilingual-asr

# 单张显卡训练
TMPDIR=/mnt/all_ssd/tmp/torch python -m workflows.recipes.wav2vec2.asr ../ft_records --config-file ../recipes/ipa_250506_260120.yaml

# 8 张显卡训练
TMPDIR=/mnt/all_ssd/tmp/torch CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 -m workflows.recipes.wav2vec2.asr ../ft_records --config-file ../recipes/ipa_250506_260120.yaml
```

一些说明：

- `TMPDIR`：fairseq2 训练过程中的临时文件默认存放在 `/tmp/tmpXXXXXXXX/`，但 `/tmp` 限定了文件夹不超过 192 MB，训练不久后临时文件就会达到这一限制，导致训练中断。现在手动指定之。由于临时文件读写频繁，设在 `all_hdd` 负载太大，所以设在 `all_ssd`
- `../ft_records`：指定输出的文件夹。微调（fine-tune，这里缩写为 `ft`）相关的各类文件会输出到 `../ft_records/ws_N.XXXXXXXX` 下，`N` 为显卡数量，`XXXXXXXX` 为根据当前 `.yaml` 配置生成的。可以通过 `ls ft_records/ -l` 查看各 `ws_N.XXXXXXXX` 文件夹的修改时期，来判断哪一个 `ws_N.XXXXXXXX` 是当前训练所用的文件夹
- 在命令前面还可以增加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 以进一步少许降低显存占用

注意训练一开始时预估的所需时间不准确，因为第一阶段的训练比第二阶段快很多，这里预估的时间是根据第一阶段的速度预估的。

### 3.4 继续微调

训练中断后，在 `.yaml` 配置文件保持不变、显卡数量不变的前提下，可以直接执行原命令，就会从上一个保存的 checkpoint 继续训练。

### 3.5 观察错误率

每次评估的单元错误率和词错误率以及其他参数保存在 `ft_records/ws_N.XXXXXXXX/metrics/valid.jsonl` 中。运行 `extract_scores.py` 可将两个错误率导出成 `.tsv`。

## 4. 使用微调后的模型

微调过程中的 checkpoint 保存在 `ft_records/ws_N.XXXXXXXX/checkpoints/step_XXXX/model/pp_00/tp_00/`。为了方便后续识别，在训练完成后可将 `ws_N.XXXXXXXX` 改名为有意义的文件夹名（如果是训练中断未完成，之后还需继续训练的话，就不要改名）。

### 4.1 合并模型文件

使用多卡微调时，checkpoint 的保存是把每张显卡上的模型权重分开、逐一保存的，在 `tp_00/` 下保存为 `sdp_00.pt` 至 `sdp_{N-1}.pt`，需要将其合并为一个模型文件才能使用。运行 `consolidate_checkpoint.py` 即可合并，把合并后的模型文件保存到 `models/` 下。

如果是使用单卡微调，`tp_00/` 下只有一个 `sdp_00.pt`。将其复制到 `models/` 下即可。

### 4.2 在本地添加模型卡片

示例见 `my_models.yaml`，与 2.6 一样修改其信息后将其复制到 `~/.config/fairseq2/assets/` 下即可。如有多个模型卡片，之间用 `---` 隔开（参考[这里](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/cards/models/rc_models_v2.yaml)配置多个模型卡片的写法）。

执行 `python -m fairseq2.assets list --kind model` 看该模型卡片的路径配置是否正确，能够列出。

### 4.3 本地使用

见 1.4 示例。如果在 2.3 使用了 `char_substitution.tsv`，那么别忘了替换回来：

```python
char_substitution_path = ...
char_substitution = {}
with open(char_substitution_path, 'r', encoding='utf-8') as f:
    for line in f:
        char_old, char_new = line.strip('\n').split('\t')
        char_substitution[char_new] = char_old

# transcriptions = pipeline.transcribe(...)

transcriptions = [''.join([char_substitution.get(c, c) for c in t])
                  for t in transcriptions]
```

### 4.4 在 Docker 部署

可参考 IPA-Transcribe v4 的[部署说明](https://github.com/EL-CL/IPA-Transcribe#deploy-in-docker-1)和 [Dockerfile](https://github.com/EL-CL/IPA-Transcribe/blob/main/v4.Dockerfile)。

## 5. 用测试集评估模型

### 5.1 自动评估

评估的配置文件例子见 `recipes/ipa_250506_260120_eval.yaml`。

```bash
cd omnilingual-asr

# 单张显卡评估
TMPDIR=/mnt/all_ssd/tmp/torch python -m workflows.recipes.wav2vec2.asr.eval ../ft_records --config-file ../recipes/ipa_250506_260120_eval.yaml

# 8 张显卡评估
TMPDIR=/mnt/all_ssd/tmp/torch CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 -m workflows.recipes.wav2vec2.asr.eval ../ft_records --config-file ../recipes/ipa_250506_260120_eval.yaml
```

### 5.2 手动评估

如果想查看每条音频的错误率，可运行 `eval_by_audio.py`。运行前需安装 `pip install levenshtein`。
