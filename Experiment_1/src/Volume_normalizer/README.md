# Volume_normalizer

----
## normalize.py
音声ファイルの最大振幅を揃える．
音声ファイル毎に正規化処理を行う．

### Usage sample
```
python normalize.py \
  --source_dir <SOURCE_DIR> \
  --max_amplitude <MAX_AMPLITUDE> \
  --replace \
  --extension <EXTENSION>
```
```
python normalize.py \
  --source_dir <SOURCE_DIR> \
  --max_amplitude <MAX_AMPLITUDE> \
  --output_dir <OUTPUT_DIR> \
  --extension <EXTENSION>
```

### Arguments
#### Required arguments
| Argument | Help |
|----------|------|
| --source_dir | 入力ファイルのルートディレクトリ |
| --replace<br>--output\_dir | 入力ファイルを置き換える<br>出力先ディレクトリ |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --max_amplitude | 30000 | 正規化後の最大振幅値 |
| --extension | wav | 対象とするファイルの拡張子 |
