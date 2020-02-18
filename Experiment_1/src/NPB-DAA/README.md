# NPB-DAA

Required: [NPB-DAA](https://github.com/RyoOzaki/npbdaa)

----
## unroll_default_config.py
デフォルトのコンフィグファイルを展開する．

### Usage sample
```
python unroll_default_config.py \
  --default_config <DEFAULT_CONFIG> \
  --output_dir <OUTPUT_DIR>
```

### Arguments

#### Required arguments
None

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --default_config | hypparams/defaults.config | 展開するデフォルトコンフィグファイル |
| --output_dir | hypparams | 展開先ディレクトリ |

### Another outputs
None

----
## train.py
NPB-DAAを学習する．

### Usage sample
```
python train.py \
  --train_data <TRAIN_DATA> \
  --model <MODEL> \
  --letter_duration <LETTER_DURATION> \
  --letter_hsmm <LETTER_HSMM> \
  --letter_observation <LETTER_OBSERVATION> \
  --pyhlm <PYHLM> \
  --word_length <WORD_LENGTH>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --train_data | 学習データ |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --model | hypparams/model.config | モデルハイパーパラメータ |
| --letter_duration | hypparams/letter_duration.config | 音素継続長ハイパーパラメータ |
| --letter_hsmm | hypparams/letter_hsmm.config | 音素HSMMハイパーパラメータ |
| --letter_observation | hypparams/letter_observation.config | 音素出力ハイパーパラメータ |
| --pyhlm | hypparams/pyhlm.config | pyhlmハイパーパラメータ |
| --word_length | hypparams/word_length.config | 単語長ハイパーパラメータ |

### Another outputs
* results
  * word_stateseq.npz
  * letter_stateseq.npz
  * word_durations.npz
* summary_files
  * log_likelihood.txt
  * resample_times.txt
* parameters
  * ITR_\*\*\*\*.npz

----
## summary.py
NPB-DAAの学習結果を評価する．

### Usage sample
```
python summary.py \
  --results_dir <RESULTS_DIR> \
  --phn_label <PHN_LABEL> \
  --wrd_label <WRD_LABEL> \
  --speaker_id <SPEAKER_ID> \
  --model <MODEL> \
  --figure_dir <FIGURE_DIR> \
  --summary_dir <SUMMARY_DIR>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --phn_label | 正解音素ラベル |
| --wrd_label | 正解単語ラベル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --speaker_id | | 話者ファイル |
| --results_dir | ./results | 分節化結果のディレクトリ |
| --model | hypparams/model.config | モデルハイパーパラメータ |
| --figure_dir | ./figures | 図出力ディレクトリ |
| --summary_dir | ./summary_files | 評価結果出力ディレクトリ |

### Another outputs
Omitted

----
## summary_summary.py
複数試行の学習結果を集約する．

### Usage sample
```
python summary_summary.py \
  --result_dir <RESULTS_DIR> \
  --figure_dir <FIGURE_DIR> \
  --summary_dir <SUMMARY_DIR>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --result_dir | 複数試行の分節化結果ディレクトリ |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --figure_dir | ./figures | 図出力ディレクトリ |
| --summary_dir | ./summary_files | 評価結果出力ディレクトリ |

### Another outputs
Omitted

----
## runner.sh
train.pyを複数試行実行する．

### Usage sample
```
bash runner.sh \
  -t <TRAIN_DATA> \
  -p <PHN_LABEL> \
  -w <WRD_LABEL> \
  -s <SPEAKER_ID> \
  -l <LABEL> \
  -b <BEGIN_INDEX> \
  -e <END_INDEX>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| -t | 学習ファイル |
| -p | 正解音素ラベル |
| -w | 正解単語ラベル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| -s | | 話者ファイル |
| -l | sample_results | 学習結果出力名 |
| -b | 1 | 学習開始試行 |
| -e | 20 | 学習終了試行 |

----
## watchdog_runner.sh
runner.shを実行しながら監視し，処理が異常停止していた場合に自動で再開する．

### Usage sample
```
bash watchdog_runner.sh \
  -t <TRAIN_DATA> \
  -p <PHN_LABEL> \
  -w <WRD_LABEL> \
  -s <SPEAKER_ID> \
  -l <LABEL> \
  -b <BEGIN_INDEX> \
  -e <END_INDEX> \
  -c
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| -t | 学習ファイル |
| -p | 正解音素ラベル |
| -w | 正解単語ラベル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| -s | | 話者ファイル |
| -l | sample_results | 学習結果出力名 |
| -b | 1 | 学習開始試行 |
| -e | 20 | 学習終了試行 |
| -c | | 指定時にcontinue.shを用いて学習再開 |

----
## clean.sh
runner.sh, watchdog_runner.sh, train.pyで作成されるディレクトリ等を削除する．

### Usage sample
```
bash clean.sh
```

### Arguments

#### Required arguments
None

#### Optional arguments
None
