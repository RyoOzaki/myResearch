# Normalize

----
## uniform_normalize.py
各次元で正規化を行う．

### Usage sample
```
python uniform_normalize.py \
  --source <SOURCE_FILE> \
  --speaker_id <SPEKAER_ID> \
  --min_value <MIN_VALUE> \
  --max_value <MAX_VALUE> \
  --output <OUTPUT>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 入力ファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --output | \[\*1\] | 出力ファイル |
| --speaker_id | | 話者ファイル \[\*2\] |
| --min_value | 0.0 | 正規化後の最小値 |
| --max_value | 1.0 | 正規化後の最大値 |

\[\*1\]:
指定が無い場合は，入力ファイル名先頭に"uninorm\_"を付したファイルに出力．
例えば，"--train_data dir1/dir2/source.npz"が指定されたとき，出力ファイルは"dir1/dir2/uninorm\_source.npz"．
ただし，"--speaker\_id"が指定されている場合は，入力ファイル名先頭に"uninorm\_spkind\_"を付したファイルに出力．
例えば，"--train_data dir1/dir2/source.npz"が指定されたとき，出力ファイルは"dir1/dir2/uninorm\_spkind\_source.npz"．

\[\*2\]:
"--speaker\_id"が指定されている場合は，話者ごとに最小値と最大値を算出し正規化を行う．

### Another outputs

 * "--speaker\id"の指定が無い場合<br>
出力ファイルと同名のディレクトリ内に"min.npy"と"max.npy"というファイル名で元データの最小値ベクトルと最大値ベクトルを出力．

* "--speaker\id"が指定されている場合<br>
出力ファイルと同名のディレクトリ内に"mins.npz"と"maxs.npz"というファイル名で各話者の最小値ベクトルと最大値ベクトルを出力．

----
## gaussian_normalize.py
各次元で標準化(平均0, 分散1)を行う．

### Usage sample
```
python gaussian_normalize.py \
  --source <SOURCE> \
  --speaker_id <SPEAKER_ID> \
  --output <OUTPUT>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 入力ファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --output | \[\*1\] | 出力ファイル |
| --speaker_id | | 話者ファイル \[\*2\] |

\[\*1\]:
指定が無い場合は，入力ファイル名先頭に"gaunorm\_"を付したファイルに出力．
例えば，"--train_data dir1/dir2/source.npz"が指定されたとき，出力ファイルは"dir1/dir2/gaunorm\_source.npz"．
ただし，"--speaker\_id"が指定されている場合は，入力ファイル名先頭に"gaunorm\_spkind\_"を付したファイルに出力．
例えば，"--train_data dir1/dir2/source.npz"が指定されたとき，出力ファイルは"dir1/dir2/gaunorm\_spkind\_source.npz"．

\[\*2\]:
"--speaker\_id"が指定されている場合は，話者ごとに各次元で標準化(平均0, 分散1)を行う．

### Another outputs

 * "--speaker\id"の指定が無い場合<br>
出力ファイルと同名のディレクトリ内に"mean.npy"と"std.npy"というファイル名で元データの平均ベクトルと標準偏差ベクトルを出力．

* "--speaker\id"が指定されている場合<br>
出力ファイルと同名のディレクトリ内に"means.npz"と"stds.npz"というファイル名で各話者の平均ベクトルと標準偏差ベクトルを出力．

----
## gaussian_normalize_with_f0.py
各次元で標準化(平均0, 分散1)を行う．
ただし，f0の値が0である点を除く．

### Usage sample
```
python gaussian_normalize_with_f0.py \
  --source <SOURCE> \
  --source_f0 <SOURCE_F0> \
  --speaker_id <SPEAKER_ID> \
  --output <OUTPUT>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 入力ファイル |
| --source_f0 | f0の入力ファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --output | \[\*1\] | 出力ファイル |
| --speaker_id | | 話者ファイル \[\*2\] |

\[\*1\]:
指定が無い場合は，入力ファイル名先頭に"gaunorm\_with\_f0\_"を付したファイルに出力．
例えば，"--train_data dir1/dir2/source.npz"が指定されたとき，出力ファイルは"dir1/dir2/gaunorm\_with\_f0\_source.npz"．
ただし，"--speaker\_id"が指定されている場合は，入力ファイル名先頭に"gaunorm\_with\_f0\_spkind\_"を付したファイルに出力．
例えば，"--train_data dir1/dir2/source.npz"が指定されたとき，出力ファイルは"dir1/dir2/gaunorm\_with\_f0\_spkind\_source.npz"．

\[\*2\]:
"--speaker\_id"が指定されている場合は，話者ごとに各次元で標準化(平均0, 分散1)を行う．

### Another outputs

 * "--speaker\id"の指定が無い場合<br>
出力ファイルと同名のディレクトリ内に"mean.npy"と"std.npy"というファイル名で元データの平均ベクトルと標準偏差ベクトルを出力．
加えて，出力ファイルと同名のディレクトリ内に"lf0_mean.npy"と"lf0_std.npy"というファイル名でlog f0の平均ベクトルと標準偏差ベクトルを出力．

* "--speaker\id"が指定されている場合<br>
出力ファイルと同名のディレクトリ内に"means.npz"と"stds.npz"というファイル名で各話者の平均ベクトルと標準偏差ベクトルを出力．
加えて，出力ファイルと同名のディレクトリ内に"lf0_means.npz"と"lf0_stds.npz"というファイル名で各話者のlog f0の平均ベクトルと標準偏差ベクトルを出力．
