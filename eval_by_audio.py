import Levenshtein as Lev
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

model_card = 'ipa_250506_260120'
metadata_path = 'datasets/ipa_250506.test.tsv'
result_path = 'ipa_250506.test.result.tsv'
char_substitution_path = 'char_substitution.tsv'
batch_size = 16  # 此设置对结果没有影响

char_substitution = {}
with open(char_substitution_path, 'r', encoding='utf-8') as f:
    for line in f:
        char_old, char_new = line.strip('\n').split('\t')
        char_substitution[char_new] = char_old

metadata = []
with open(metadata_path, 'r', encoding='utf-8') as f:
    for line in f:
        path, text = line.strip('\n').split('\t')
        metadata.append((path, text))
# 按文件名（path）排序
metadata.sort(key=lambda x: x[0])

# 推理
pipeline = ASRInferencePipeline(model_card=model_card)
paths = [line[0] for line in metadata]
ref_texts = [line[1] for line in metadata]
texts = pipeline.transcribe(
    paths, lang=['ipa_Latn'] * len(paths), batch_size=batch_size)

# 计算错误率
result = []
for path, ref_text, text in zip(paths, ref_texts, texts):
    uer = Lev.distance(text, ref_text) / len(ref_text) * 100
    result.append([
        path,
        ''.join([char_substitution.get(c, c) for c in ref_text]),
        ''.join([char_substitution.get(c, c) for c in text]),
        uer,
    ])
uers = [line[-1] for line in result]
print(f'Average UER: {sum(uers) / len(uers)}')

with open(result_path, 'w', encoding='utf-8') as f:
    for line in result:
        f.write('\t'.join([str(x) for x in line]) + '\n')
