# Evaluate_summary

----
## pickup_all_result.py
入力ファイルの分布を表示する．

### Usage sample
```
python pickup_all_result.py \
  --result_dir <RESULT_DIR> \
  --output_dir <OUTPUT_DIR>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --result_dir | NPB-DAAの分節化結果が配置されたディレクトリ |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --output_dir | ./summary | summary fileの出力ディレクトリ |

### Another outputs
None

----
## show_results.py
PCAを用いて2次元に削減後，音素および話者ごとに散布図を作成．

### Usage sample
```
python show_results.py \
  --result_dir <RESULT_DIR> \
  --score <SCORE> \
  --keyword <KEYWORD> \
  --exclude_keyword <EXCLUDE_KEYWORD> \
  --iteration <ITERATION> \
  --with_max \
  --with_min \
  --with_map \
  --with_corr \
  --float_format <FLOAT_FORMAT> \
  --pm_str <PM_STR> \
  --prefix <PREFIX> \
  --suffix <SUFFIX>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --result_dir | pickup_all_result.pyの出力先ディレクトリ |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --score | letter_ARI word_ARI | 出力スコア [\*1] |
| --keyword | | 表示する結果に含まれる文字列 |
| --exclude_keyword | | 表示しない結果に含まれる文字列 |
| --iteration | -1 | スコアを表示するイテレーション |
| --with_max | | 最大値も併せて表示 |
| --with_min | | 最小値も併せて表示 |
| --with_map | | 最尤値も併せて表示 |
| --with_corr | | log-likelihoodsとの相関係数も併せて表示 |
| --float_format | .3f | 小数表示のフォーマット |
| --pm_str | ± | プラスマイナスに用いる文字列 |
| --prefix | | 平均と標準偏差の表示の際に先頭に追加する文字列 |
| --suffix | | 平均と標準偏差の表示の際に末尾に追加する文字列 |

[\*1]:
指定可能なスコアは次の通りである．
* letter_ARI
* letter_micro_F1
* letter_macro_F1
* word_ARI
* word_micro_F1
* word_macro_F1
* resample_times
* log_likelihoods

### Another outputs
None

----
## plot_segmentation_result.py
NPB-DAAの分節化結果を保存する．

### Usage sample
```
python plot_segmentation_result.py \
  --result_dir <RESULT_DIR> \
  --model <MODEL> \
  --word_stateseq <WORD_STATESEQ> \
  --letter_stateseq <LETTER_STATESEQ> \
  --word_durations <WORD_DURATIONS> \
  --phn <PHN> \
  --wrd <WRD> \
  --Ft_phn <FT_PHN> \
  --Ft_wrd <FT_WRD> \
  --figure_dir <FIGURE_DIR> \
  --figsize <FIGSIZE> \
  --keep_dir
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --result_dir | NPB-DAAの分節化結果が配置されたディレクトリ [\*1] |
| --word_stateseq | 単語分節化結果ファイル [\*1] |
| --letter_stateseq | 音素分節化結果ファイル [\*1] |
| --word_durations | 単語長分節化結果ファイル [\*1] |
| --phn | 音素ラベルファイル |
| --wrd | 単語ラベルファイル |
| --Ft_phn | 音素切り替わりラベルファイル |
| --Ft_wrd | 単語切り替わりラベルファイル |
| --figure_dir | 出力ディレクトリ |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --model | hypparams/model.config | NPB-DAAのmodelハイパーパラメータを記述したファイル |
| --figsize | 16 8 | 出力画像サイズ |
| --keep_dir | | 分節化結果の出力の際にディレクトリ構造を保つ |

[\*1]:
"--result_dir"の指定がある場合，指定されたディレクトリ内の"word_stateseq.npz", "letter_stateseq.npz", "word_durations.npz"が自動的に用いられる．

### Another outputs
None
