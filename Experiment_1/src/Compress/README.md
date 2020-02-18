# Compress

----
## PCA_compress.py
PCAで次元数を削減．

### Usage sample
```
python PCA_compress.py \
  --source <SOURCE_FILE> \
  --output <OUTPUT_FILE> \
  --n_components <N_COMPONENTS>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 入力ファイル |
| --n_components | 削減後の次元数 |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --output | \[\*1\] | 出力ファイル |

\[\*1\]:
指定が無い場合は，入力ファイル名先頭に"pca\_"を付したファイルに出力．
例えば，"--source dir1/dir2/source.npz"が指定されたとき，出力ファイルは"dir1/dir2/pca\_source.npz"．

### Another outputs
出力ファイルと同名のディレクトリ内に"components.npy"というファイル名で主成分行列を出力．
