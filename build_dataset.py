from functools import partial
import numpy as np
import pyarrow as pa
import soundfile as sf
import ray
from datasets import Dataset, DatasetDict

metadata_path = 'datasets/ipa_250506.tsv'
char_substitution_path = 'char_substitution.tsv'
output_dir = 'datasets/ipa_250506'
dataset_card_name = 'ipa_250506'
lang_id = 'ipa_Latn'
dev_size = 0.01  # 取 1% 作为验证集
test_size = 0.01  # 取 1% 作为测试集
seed = 542  # 使用固定的随机数种子拆分数据集，以确保每次拆分结果一致
num_cpus = 60  # 线程数
num_audios_per_shard = 6000  # 导出时，每个分片包含的音频数量


# 元数据转换为 Hugging Face 数据集格式，并且拆分成 train、dev、test
def metadata_to_hf(paths, texts):
    dataset = Dataset.from_dict({
        'path': paths,
        'text': texts,
    })

    # 只拆分成 train 和 dev 的情况
    if test_size == 0:
        train_dev = dataset.train_test_split(
            dev_size, seed=seed)
        return DatasetDict({
            'train': train_dev['train'],
            'dev': train_dev['test'],
        })

    # 拆分成 train、dev 和 test 的情况
    # 先拆分成 train+dev 和 test
    train_test = dataset.train_test_split(
        test_size, seed=seed)
    # 再拆分成 train 和 dev
    train_dev = train_test['train'].train_test_split(
        dev_size / (1 - test_size), seed=seed)
    return DatasetDict({
        'train': train_dev['train'],
        'dev': train_dev['test'],
        'test': train_test['test']
    })


# 把二进制音频数据转为 int8 列表
# 摘自 omnilingual_asr/datapre/audio_tools.py
def binary_to_list_int8(binary_array: pa.Array | pa.ChunkedArray) -> pa.Array:
    '''
    Efficiently convert a pyarrow BinaryArray to a ListArray of int8.
    Each binary value becomes a list of int8 values (that's copy-less method)
    Nulls are preserved.
    '''
    if not pa.types.is_binary(binary_array.type):
        raise ValueError('Input array must be of binary type.')
    if isinstance(binary_array, pa.ChunkedArray):
        binary_array = binary_array.combine_chunks()

    # Get buffers: [null_bitmap, offsets, data]
    buffers = binary_array.buffers()
    offsets = buffers[1]
    data = buffers[2]
    offset = binary_array.offset

    # Offsets as numpy array
    offsets_np = np.frombuffer(offsets, dtype='int32')[  # type: ignore
        offset: offset + len(binary_array) + 1
    ]

    data_np = np.frombuffer(data, dtype='int8')[offsets_np[0]:]  # type: ignore
    offsets_np -= offsets_np[0]
    values_array = pa.array(data_np, type=pa.int8())

    list_array = pa.ListArray.from_arrays(
        offsets_np, values_array, mask=binary_array.is_null()
    )
    return list_array


# 把 Hugging Face 格式转换为 Omnilingual ASR 所需格式
def convert_batch(batch, corpus, split, language):
    audio_size_list = []
    audio_bytes_list = []
    for path in batch['path'].to_pylist():
        info = sf.info(path)
        assert info.samplerate == 16000
        audio_size = info.frames
        audio_size_list.append(audio_size)

        with open(path, 'rb') as f:
            audio_bin = f.read()
        audio_bin = pa.array([audio_bin], type=pa.binary())
        audio_bytes = binary_to_list_int8(audio_bin).to_pylist()[0]
        audio_bytes_list.append(audio_bytes)
    return batch \
        .drop(['path']) \
        .append_column('audio_size', pa.array(audio_size_list, type=pa.int64())) \
        .append_column('audio_bytes', pa.array(audio_bytes_list, type=pa.list_(pa.int8()))) \
        .append_column('corpus', pa.array([corpus] * len(batch), type=pa.string())) \
        .append_column('split', pa.array([split] * len(batch), type=pa.string())) \
        .append_column('language', pa.array([language] * len(batch), type=pa.string()))


char_substitution = {}
with open(char_substitution_path, 'r', encoding='utf-8') as f:
    for line in f:
        char_old, char_new = line.strip('\n').split('\t')
        char_substitution[char_old] = char_new
paths = []
texts = []
with open(metadata_path, 'r', encoding='utf-8') as f:
    for line in f:
        path, text = line.strip('\n').split('\t')
        text = ''.join([char_substitution.get(c, c) for c in text])
        paths.append(path)
        texts.append(text)
datasets = metadata_to_hf(paths, texts)
for split, dataset in datasets.items():
    print(f'{split}: {len(dataset)} samples')
print()


ray.init()
for split, dataset in datasets.items():
    print(f'==== Start processing split: {split} ====')
    ray_stream = ray.data.from_huggingface(dataset)
    ray_stream = ray_stream.repartition(num_blocks=num_cpus * 4)
    ray_stream = ray_stream.map_batches(
        partial(
            convert_batch,
            corpus=dataset_card_name,
            split=split,
            language=lang_id,
        ),
        batch_size=100,
        batch_format='pyarrow',
        concurrency=num_cpus,
    )
    ray_stream.write_parquet(
        output_dir,
        partition_cols=['corpus', 'split', 'language'],
        min_rows_per_file=num_audios_per_shard,
        row_group_size=100,
    )
    print(f'==== Finished processing split: {split} ====')
    print()
