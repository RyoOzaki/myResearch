# DSAE-PBHL

Required: [DSAE-PBHL](https://github.com/RyoOzaki/DSAE-PBHL)

----
## DSAE_train.py
DSAEを学習し，次元数を削減．

### Usage sample
```
python DSAE_train.py \
  --structure <STRUCTURE> \
  --train_data <TRAIN_FILE> \
  --output <OUTPUT_FILE> \
  --epoch <EPOCH> \
  --threshold <THRESHOLD>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --train_data | 学習ファイル |
| --structure | ネットワーク構造 \[\*1\] |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --output | \[\*2\] | 出力ファイル |
| --epoch | 10 | エポック数 (thresholdによる終了判定がエポック毎に行われる) |
| --threshold | 1.0E-60 | Lossの値の変化がthreshold以下で計算打ち切り |

\[\*1\]:
ネットワーク構造の記述は，整数値を並べて行う．
例えば，20次元の入力に対して，順に20 -> 10 -> 5 -> 3次元に削減する場合は，"--structure 20 10 5 3"を指定する．

\[\*2\]:
指定が無い場合は，入力ファイル名先頭に"dsae\_"を付したファイルに出力．
例えば，"--train_data dir1/dir2/source.npz"が指定されたとき，出力ファイルは"dir1/dir2/dsae\_source.npz"．

### Another outputs
出力ファイルと同名のディレクトリ内に"dsae.npz"というファイル名でDSAEのパラメータを出力．

----
## DSAE_PBHL_train.py
DSAE-PBHLを学習し，次元数を削減．

### Usage sample
```
python DSAE_PBHL_train.py \
  --structure <STRUCTURE> \
  --pb_structure <PB_STRUCTURE> \
  --train_data <TRAIN_FILE> \
  --speaker_id <SPEAKER_ID> \
  --output <OUTPUT_FILE> \
  --epoch <EPOCH> \
  --threshold <THRESHOLD>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --train_data | 学習ファイル |
| --speaker_id | 話者ファイル |
| --structure | ネットワーク構造 \[\*1\] |
| --pb_structure_aaa | パラメトリックバイアスのネットワーク構造 \[\*2\] |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --output | \[\*3\] | 出力ファイル |
| --epoch | 10 | エポック数 (thresholdによる終了判定がエポック毎に行われる) |
| --threshold | 1.0E-60 | Lossの値の変化がthreshold以下で計算打ち切り |

\[\*1\]:
ネットワーク構造の記述は，整数値を並べて行う．
例えば，20次元の入力に対して，順に20 -> 10 -> 5 -> 3次元に削減する場合は，"--structure 20 10 5 3"を指定する．

\[\*2\]:
パラメトリックバイアスのネットワーク構造の記述は，structureの指定と同様に整数値を並べて行う．
例えば，4次元の入力に対して，中間層でのパラメトリックバイアスの次元数を3次元とする場合は，"--pb_structure 4 3"を指定する．


\[\*3\]:
指定が無い場合は，入力ファイル名先頭に"dsae\_pbhl\_"を付したファイルに出力．
例えば，"--train_data dir1/dir2/source.npz"が指定されたとき，出力ファイルは"dir1/dir2/dsae\_pbhl\_source.npz"．

### Another outputs
出力ファイルと同名のディレクトリ内に"dsae.npz"というファイル名でDSAEのパラメータを出力．
同様に"pb\_means.npz"というファイル名で，パラメトリックバイアスを話者毎に平均したベクトルを出力．
