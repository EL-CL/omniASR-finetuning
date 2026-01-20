import os
import sentencepiece as spm

model_path = '~/.cache/fairseq2/assets/e7be1a6acb8f76fdbca19dce/omniASR_tokenizer_written_v2.model'
model_path = os.path.expanduser(model_path)
output_path = 'omniASR_tokenizer_written_v2_vocabulary.txt'

sp = spm.SentencePieceProcessor(model_file=model_path)
vocab_size = sp.get_piece_size()
tokens = [sp.id_to_piece(i) for i in range(vocab_size)]

print(f"Vocabulary Size: {vocab_size}")
with open(output_path, 'w', encoding='utf-8') as f:
    for token in tokens:
        f.write(token + '\n')
