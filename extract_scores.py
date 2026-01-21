import json

jsonl_path = 'ft_records/ipa_250506_260120/metrics/valid.jsonl'
output_path = 'scores.tsv'

lines = []
with open(jsonl_path, encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        line = (
            line['Step'],
            line['Unit Error Rate (UER)'],
            line['Word Error Rate (WER)'],
        )
        lines.append(line)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write('Step\tUER\tWER\n')
    for step, uer, wer in lines:
        f.write(f'{step}\t{uer}\t{wer}\n')
