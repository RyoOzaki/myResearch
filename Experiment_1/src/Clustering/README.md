# Clustering

----
## Kmeans.py
K-meansを用いたクラスタリングを行う．

### Usage sample
```
python Kmeans.py \
  --source <SOURCE_FILE> \
  --phn <PHN_FILE> \
  --K <NUM_OF_CLASS>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 入力ファイル |
| --phn | 音素のGrand truth |
| --K | クラス数 |

#### Optional arguments
None

----
## GMM_Gibbs.py
GMMをGibbs samplingを用いてクラスタリングを行う．
ハイパーパラメータはすべてソースコード内に記述．

### Usage sample
```
python GMM_Gibbs.py \
  --source <SOURCE_FILE> \
  --phn <PHN_FILE> \
  --K <NUM_OF_CLASS> \
  --print_all_ARI \
  --trial <TRIAL>
```
```
python GMM_Gibbs.py \
  --source <SOURCE_FILE> \
  --phn <PHN_FILE> \
  --K <NUM_OF_CLASS> \
  --print_ARI_span <PRINT_ARI_SPAN> \
  --trial <TRIAL>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 入力ファイル |
| --phn | 音素のGrand truth |
| --K | クラス数 |
| --iter | Gibbs samplingのイテレーション回数 |


#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --print_all_ARI | | 全イテレーションにおいてARI値を出力 |
| --print_ARI_span | | ARIを指定イテレーション毎に出力 |
| --trial | 1 | 試行回数 |

----
## GMM_EM.py
GMMをEM algorithmを用いてクラスタリングを行う．

### Usage sample
```
python GMM_EM.py \
  --source <SOURCE_FILE> \
  --phn <PHN_FILE> \
  --n_components <N_COMPONENTS> \
  --trial <TRIAL>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 入力ファイル |
| --phn | 音素のGrand truth |
| --n_components | クラス数 |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --trial | 1 | 試行回数 |
