# LSTMLM (LSTM language model)

----
## convert.py
LSTM-LMの学習用にNPB-DAAの分節化結果を編集する．

### Usage sample
```
python convert.py \
  --word_num <WORD_NUM> \
  --output <OUTPUT> \
  --word_stateseq <WORD_STATESEQ> \
  --word_durations <WORD_DURATIONS> \
  --using_iterations <USING_ITERATIONS>
```

### Arguments
#### Required arguments
| Argument | Help |
|----------|------|
| --word_num | 単語数 |
| --word_stateseq | NPB-DAAの単語分節化結果ファイル |
| --word_durations | NPB-DAAの単語継続長結果ファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --output | sentences.npz | センテンスファイルの出力先 |
| --using_iterations | -1 | 使用するNPB-DAAのイテレーション (複数指定可) |

### Another outputs
None

----
## LSTMLM_train.py
LSTM-LMの学習を行う．

### Usage sample
```
python LSTMLM_train.py \
  --model <MODEL> \
  --output_model <OUTPUT_MODEL> \
  --train_data <TRAIN_DATA> \
  --train_iter <TRAIN_ITER> \
  --hidden_node <HIDDEN_NODE> \
  --init_model
```

### Arguments
#### Required arguments
| Argument | Help |
|----------|------|
| --model | 読み込み・保存するLSTM-LMのモデルファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --train_data | sentences.npz | 学習ファイル |
| --train_iter | 100 | イテレーション数 |
| --hidden_node | 128 | LSTMの中間層の次元数 |
| --output_model | | "--model"と別ファイルとして保存する |
| --init_model | | "--model"を読み込まずに初期化した状態から学習を開始する |

### Another outputs
None
