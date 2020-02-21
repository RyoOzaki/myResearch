# StarGAN-VC

----
## train_stargan-vc.py
StarGAN-VCの学習を行う．

### Usage sample
```
python train_stargan-vc.py \
  --train_data <TRAIN_DATA> \
  --speaker_id <SPEAKER_ID> \
  --output_file <OUTPUT_FILE> \
  --epoch <EPOCH> \
  --epoch_start <EPOCH_START> \
  --snapshot <SNAPSHOT> \
  --batchsize <BATCHSIZE> \
  --optimizer <OPTIMIZER> \
  --lrate <LEARNING_RATE> \
  --genpath <GENERATOR_PATH> \
  --clspath <CLASSIFIER_PATH> \
  --advdispath <DISCRIMINATOR_PATH> \
  --gpu <GPU>
```

### Arguments
#### Required arguments
| Argument | Help |
|----------|------|
| --train_data | 学習ファイル |
| --speaker_id | 話者ファイル |
| --output_file | 出力ファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --epoch | 6000 | 学習エポック数 |
| --epoch_start | 0 | 学習エポック開始数 |
| --snapshot | 100 | スナップショット保存エポック数 |
| --batchsize | 4 | バッチサイズ |
| --optimizer | Adam | オプティマイザ (Adam / MomentumSGD / RMSprop) |
| --lrate | 0.00001 | 学習率 ("--optimizer Adam"は関係なし) |
| --genpath |  | Generatorのパス (学習再開等で利用) |
| --clspath |  | Classifierのパス (学習再開等で利用) |
| --advdispath |  | Discriminatorのパス (学習再開等で利用) |
| --gpu | -1 | 使用するGPU番号 (-1でCPUのみ使用) |

### Another outputs
"--output_file dir1/dir2/out.npz"が指定された場合
* dir1/dir2/out/sgvc_log/
  * Lossの値をテキストファイルで出力
* dir1/dir2/out/snapshot/
  * snapshotを保存
* dir1/dir2/out/snapshot_feature/
  * 中間層の値をsnapshotと同じ頻度で保存

----
## train_stargan-vc_new.py
StarGAN-VC_newの学習を行う．
修論での使用は無し．
GeneratorのEncoderに入力元話者の情報が入力されたバージョン

### Usage sample
```
python train_stargan-vc_new.py \
  --train_data <TRAIN_DATA> \
  --speaker_id <SPEAKER_ID> \
  --output_file <OUTPUT_FILE> \
  --epoch <EPOCH> \
  --epoch_start <EPOCH_START> \
  --snapshot <SNAPSHOT> \
  --batchsize <BATCHSIZE> \
  --optimizer <OPTIMIZER> \
  --lrate <LEARNING_RATE> \
  --genpath <GENERATOR_PATH> \
  --clspath <CLASSIFIER_PATH> \
  --advdispath <DISCRIMINATOR_PATH> \
  --gpu <GPU>
```

### Arguments
#### Required arguments
| Argument | Help |
|----------|------|
| --train_data | 学習ファイル |
| --speaker_id | 話者ファイル |
| --output_file | 出力ファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --epoch | 6000 | 学習エポック数 |
| --epoch_start | 0 | 学習エポック開始数 |
| --snapshot | 100 | スナップショット保存エポック数 |
| --batchsize | 4 | バッチサイズ |
| --optimizer | Adam | オプティマイザ (Adam / MomentumSGD / RMSprop) |
| --lrate | 0.00001 | 学習率 ("--optimizer Adam"は関係なし) |
| --genpath |  | Generatorのパス (学習再開等で利用) |
| --clspath |  | Classifierのパス (学習再開等で利用) |
| --advdispath |  | Discriminatorのパス (学習再開等で利用) |
| --gpu | -1 | 使用するGPU番号 (-1でCPUのみ使用) |

### Another outputs
"--output_file dir1/dir2/out.npz"が指定された場合
* dir1/dir2/out/sgvc_log/
  * Lossの値をテキストファイルで出力
* dir1/dir2/out/snapshot/
  * snapshotを保存
* dir1/dir2/out/snapshot_feature/
  * 中間層の値をsnapshotと同じ頻度で保存

----
## test_StarGAN_VC.py
StarGAN-VCの音声変換をテスト．

### Usage sample
```
python test_StarGAN_VC.py \
  --snapshot_dir <SNAPSHOT_DIR> \
  --snapshot_name <SNAPSHOT_NAME> \
  --generator <GENERATOR> \
  --discriminator <DISCRIMINATOR> \
  --classifier <CLASSIFIER> \
  --speaker_id <SPEAKER_ID> \
  --mcep <MCEP> \
  --f0 <F0> \
  --ap <AP> \
  --samplerate <SAMPLERATE> \
  --fftsize <FFTSIZE> \
  --frame_period <FRAME_PERIOD> \
  --mcep_norm_param <MCEP_NORM_MEAN> <MCEP_NORM_STD> \
  --logf0_norm_param <LOGF0_NORM_MEAN> <LOGF0_NORM_STD> \
  --output_dir <OUTPUT_DIR> \
  --flatten_dir
```

### Arguments
#### Required arguments
| Argument | Help |
|----------|------|
| --snapshot_dir | スナップショットが保存されているディレクトリ\[\*1\] |
| --snapshot_name | スナップショット名 \[\*1\] |
| --generator | Generatorのパス \[\*1\] |
| --discriminator | Discriminatorのパス \[\*1\] |
| --classifier | Classifierのパス \[\*1\] |
| --speaker_id | 話者ファイル |
| --mcep | メルケプストラムファイル |
| --f0 | F0ファイル |
| --ap | 非周期性指標ファイル |
| --mcep_norm_param | メルケプストラムの平均・標準偏差ファイル |
| --logf0_norm_param | log F0の平均・標準偏差ファイル |
| --output_dir | 出力ディレクトリ |


#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --samplerate | 48000 | サンプリング周波数 |
| --fftsize | 1024 | FFTサイズ |
| --frame_period | 0.005 | 窓・シフト長 |
| --flatten_dir | | ディレクトリ構造を保たず出力 |

\[\*1\]:
引数は以下の何通りかが利用できる．
* "--generator", "--classifier", "--discriminator"の3つを指定
* "--snapshot_dir", "snapshot_name"の2つを指定

### Another outputs
None

----
## plot_losses.py
StarGAN-VCのLossをプロット．

### Usage sample
```
python plot_losses.py \
  --logdir <LOGDIR> \
  --dot <DOT> \
  --begin_iter <BEGIN_ITER>
```

### Arguments
#### Required arguments
| Argument | Help |
|----------|------|
| --logdir | Lossの出力ディレクトリ (sgvc_log) |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --dot | -1 | プロットした学習曲線に赤点でドットを描くイテレーション |
| --begin_iter | 0 | プロットし始めるイテレーション |

### Another outputs
None
