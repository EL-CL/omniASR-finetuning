import polars as pl
import pyarrow.dataset as pa_ds

parquet_dataset_root = 'datasets/ipa_250506'
output_path = 'datasets/ipa_250506_stats.tsv'

table = pa_ds.dataset(
    parquet_dataset_root, partitioning='hive', exclude_invalid_files=True
).to_table(columns=['language', 'corpus', 'audio_size'])
pl_table = pl.from_arrow(table.combine_chunks())
assert isinstance(pl_table, pl.DataFrame)
stats = pl_table.group_by(['corpus', 'language']).agg(
    (pl.col('audio_size').sum() / 3600 / 16_000).alias('hours')
)
stats.write_csv(output_path, separator='\t')
