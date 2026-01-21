import os
import torch

checkpoint_folder = 'ft_records/ipa_250506_260120/checkpoints/step_29000/model/pp_00/tp_00/'
output_path = 'ft_models/ipa_250506_260120_128x29000.pt'


print('开始加载分片')
shards = []
for i, filename in enumerate([filename for filename in sorted(os.listdir(checkpoint_folder))
                              if filename.endswith('.pt')]):
    shard_path = os.path.join(checkpoint_folder, filename)
    shard = torch.load(shard_path)
    shards.append(shard)
    print(f'第 {i} 个分片加载完成')


# 检查各分片所含 key 是否一致
key_occurrences = {}
for i, shard in enumerate(shards):
    for k in shard:
        if k not in key_occurrences:
            key_occurrences[k] = []
        key_occurrences[k].append(i)
print(f'总 key 数量: {len(key_occurrences)}')
is_consistent = True
for k, occurrence in key_occurrences.items():
    if len(occurrence) < len(shards):
        is_consistent = False
        print(f'  {k}: {occurrence}')
if is_consistent:
    print('各分片所含 key 完全一致')


print('开始合并分片')
consolidated = {}
for k in key_occurrences:
    tensors = []
    for shard in shards:
        if k in shard:
            tensors.append(shard[k])
    if len(tensors) == 1:
        consolidated[k] = tensors[0]
    else:
        consolidated[k] = torch.cat(tensors, dim=0)
print('合并完成')
torch.save(consolidated, output_path)
print('保存完成')
