import os
import sentencepiece as spm

model_path = '~/.cache/fairseq2/assets/e7be1a6acb8f76fdbca19dce/omniASR_tokenizer_written_v2.model'
model_path = os.path.expanduser(model_path)
metadata_path = 'datasets/ipa_250506.tsv'
substitution_path = 'char_substitution.tsv'

chars = set()
with open(metadata_path, 'r', encoding='utf-8') as f:
    for line in f:
        chars.update(line.strip('\n').split('\t')[1])

char_substitution = {}
with open(substitution_path, 'r', encoding='utf-8') as f:
    for line in f:
        char_old, char_new = line.strip('\n').split('\t')
        char_substitution[char_old] = char_new

sp = spm.SentencePieceProcessor(model_file=model_path)
vocab_size = sp.get_piece_size()
tokens = [sp.id_to_piece(i) for i in range(vocab_size)]

for char in sorted(chars):
    if char not in tokens:
        print(f'Out-of-vocabulary token: {char}', end='')
        if char in char_substitution:
            char = char_substitution[char]
            print(f' -> Substitution: {char}', end='')
            if char in tokens:
                print(' Solved', end='')
            else:
                print(' Still out-of-vocabulary', end='')
        print()
print('Done')
