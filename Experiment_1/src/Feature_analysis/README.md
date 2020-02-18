# Feature_analysis

----
## analyze.py
入力ファイルの分布を表示する．

### Usage sample
```
python analyze.py \
  --source <SOURCE_FILE> \
  --speaker_id <SPEKAER_ID> \
  --phn <PHN> \
  --mode <MODE> \
  --with_pca \
  --n_components <N_COMPONENTS> \
  --cmap <CMAP> \
  --bins <BINS> \
  --figsize <FIGSIZE> \
  --hist_alpha <HIST_ALPHA> \
  --data_alpha <DATA_ALPHA>
```
```
python analyze.py \
  --source <SOURCE_FILE> \
  --speaker_id <SPEKAER_ID> \
  --phn <PHN> \
  --mode <MODE> \
  --cmap <CMAP> \
  --bins <BINS> \
  --figsize <FIGSIZE> \
  --alpha <ALPHA> \
  --savefig
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 入力ファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --speaker_id | | 話者ファイル |
| --phn | | 音素ファイル |
| --mode | | none / phn / spk のうちどれか [\*1] |
| --with_pca | | PCAを用いて次元圧縮を行う |
| --n_components | | PCAを用いた次元圧縮における次元数 |
| --cmap | tab10 | matplotlibのカラーマップ |
| --bins | | ヒストグラム表示の際の分割数 |
| --figsize | 12 8 | matplotlibのfigsize |
| --alpha | 0.3 | ヒストグラムおよび散布図の各点のアルファ値 |
| --hist_alpha | --alpha | ヒストグラムのアルファ値 |
| --data_alpha | --alpha | 散布図の各点のアルファ値 |

[\*1]:
指定が無い場合，自動的に判断される．
ただし，"--phn"と"--speaker\_id"が同時に指定された場合，コンソール上にてどのモードで表示するかを入力する必要がある．

### Another outputs
None

----
## plot_pca.py
PCAを用いて2次元に削減後，音素および話者ごとに散布図を作成．

### Usage sample
```
python plot_pca.py \
  --source <SOURCE> \
  --phn <PHN> \
  --speaker_id <SPEAKER_ID> \
  --phn_list <PHN_LIST> \
  --not_show <NOT_SHOW> \
  --alpha <ALPHA> \
  --cmap <CMAP> \
  --swap_second \
  --output_dir <OUTPUT_DIR> \
  --format <FORMAT>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 入力ファイル |
| --speaker_id | 話者ファイル |
| --phn | 音素ファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --output_dir | [\*1] | 出力ディレクトリ |
| --format | pdf | 出力の拡張子 |
| --phn_list | | 図に表示する音素ラベル |
| --not_show | | --phn_listのうち表示しない音素ラベル |
| --alpha | 0.3 | 散布図の各点のアルファ値 |
| --cmap | tab10 | matplotlibのカラーマップ |
| --swap_second | | y軸をx軸に対して反転 (y = -y) |

[\*1]:
指定が無い場合は，入力ファイル名先頭に"figure"ディレクトリに入力ファイルの名前のディレクトリになる．

### Another outputs
None
