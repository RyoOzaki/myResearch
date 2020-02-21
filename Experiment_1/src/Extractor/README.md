# Extractor

Required: [python_speech_features](https://github.com/jameslyons/python_speech_features)<br>
Required: [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)

----
## extract.py
音声ファイルから特徴量を抽出する．

### Usage sample
```
python extract.py \
  --source_dir <SOURCE_DIR> \
  --extension <EXTENSION> \
  --format <FORMAT> \
  --feature_type <FEATURE_TYPE> \
  --samplerate <SAMPLERATE> \
  --winlen <WINLEN> \
  --winstep <WINSTEP> \
  --numcep <NUMCEP> \
  --nfilt <NFILT> \
  --nfft <NFFT> \
  --lowfreq <LOWFREQ> \
  --highfreq <HIGHFREQ> \
  --preemph <PREEMPH> \
  --ceplifter <CEPLIFTER> \
  --appendEnergy <APPEND_ENERGY> \
  --winfunc <WINFUNC> \
  --delta_winlen <DELTA_WINLEN> \
  --label_format <LABEL_FORMAT> \
  --phn_label_extension <PHN_LABEL_EXTENSION> \
  --wrd_label_extension <WRD_LABEL_EXTENSION> \
  --sil_label <SIL_LABEL>
```

### Arguments

#### Required arguments
None

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --source_dir | ./ | 対象ファイルのルートディレクトリ |
| --format | wave | wave / sph のうちどれか |
| --extension | wav | 対象ファイルの拡張子 |
| --feature_type | mfcc | all / mfcc / mcep / mspec / logmspec のうちどれか (複数選択可，allはすべて) |
| --samplerate | \[\*1\] | サンプリング周波数 |
| --winlen | 0.025 | ウィンドウ幅 (sec) |
| --winstep | 0.01 | シフト幅 (sec) |
| --numcep | 13 | ケプストラムの次元数 |
| --nfilt | 26 | フィルタバンク数 |
| --nfft | \[\*2\] | FFTサイズ |
| --lowfreq | 0 | 最低周波数 |
| --highfreq |  | 最高周波数 |
| --preemph | 0.97 | プリエンファシス係数 |
| --ceplifter | 22 | リフタ数 |
| --appendEnergy | False | 0次をエネルギーに置き換える |
| --winfunc | hamming | 時間窓 (hamming / none) |
| --delta_winlen | 2 | 微分近似で用いる点数 |
| --label_format | none | none / time / wave_frame / mfcc_frame のうちどれか |
| --phn_label_extension | | 音素ラベルの拡張子 |
| --wrd_label_extension | | 単語ラベルの拡張子 |
| --sil_label | | 無音を表すラベル |

\[\*1\]:
指定が無い場合は対象ファイル自体のサンプリング周波数．

\[\*2\]:
指定が無い場合は"--winlen"と同じ

### Another outputs
None

----
## extract_world.py
音声ファイルから特徴量を抽出する．

### Usage sample
```
python extract_world.py \
  --source_dir <SOURCE_DIR> \
  --extension <EXTENSION> \
  --format <FORMAT> \
  --feature_type <FEATURE_TYPE> \
  --samplerate <SAMPLERATE> \
  --frame_period <FRAME_PERIOD> \
  --numcep <NUMCEP> \
  --nfilt <NFILT> \
  --fftsize <FFT_SIZE> \
  --delta_winlen <DELTA_WINLEN> \
  --label_format <LABEL_FORMAT> \
  --phn_label_extension <PHN_LABEL_EXTENSION> \
  --wrd_label_extension <WRD_LABEL_EXTENSION> \
  --sil_label <SIL_LABEL>
```

### Arguments

#### Required arguments
None

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --source_dir | ./ | 対象ファイルのルートディレクトリ |
| --format | wave | wave / sph のうちどれか |
| --extension | wav | 対象ファイルの拡張子 |
| --feature_type | mfcc | all / mfcc / mcep / spenv / f0 / ap のうちどれか (複数選択可，allはすべて) |
| --samplerate | \[\*1\] | サンプリング周波数 |
| --frame_period | 0.005 | ウィンドウ幅・シフト幅 (sec) |
| --numcep | 13 | ケプストラムの次元数 |
| --nfilt | 26 | フィルタバンク数 |
| --fftsize | 1024 | FFTサイズ |
| --delta_winlen | 2 | 微分近似で用いる点数 |
| --label_format | none | none / time / wave_frame / mfcc_frame のうちどれか |
| --phn_label_extension | | 音素ラベルの拡張子 |
| --wrd_label_extension | | 単語ラベルの拡張子 |
| --sil_label | | 無音を表すラベル |

\[\*1\]:
指定が無い場合は対象ファイル自体のサンプリング周波数．

### Another outputs
None

----
## collect2npz.py
extract.py / extract_world.pyで抽出した特徴量ファイルをnpzファイルにまとめる．

### Usage sample
```
python collect2npz.py \
  --source_dir <SOURCE_DIR> \
  --output_dir <OUTPUT_DIR> \
  --collect_extensions <COLLECT_EXTENSIONS> \
  --with_speaker_id \
  --speaker_dir_layer <SPEAKER_DIR_LAYER>
```

### Arguments

#### Required arguments
None

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --source_dir | ./ | 対象ファイルのディレクトリ |
| --output_dir | --source_dir | 出力ディレクトリ |
| --collect_extensions | \[\*1\] | npzファイルに集めるファイル拡張子 |
| --with_speaker_id | | 話者情報を持つnpzファイルも同時に作成する |
| --speaker_dir_layer | 1 | --source_dirからみた話者ディレクトリの相対位置 \[\*2\] |

\[\*1\]:
デフォルト値に以下の拡張子が指定されている．
* mfcc
* dmfcc
* ddmfcc
* mcep
* mspec
* logmspec
* spenv
* f0
* ap
* phn
* Ft_pnh
* wrd
* Ft_wrd

\[\*2\]:
--source\_dirを0とした時の相対位置で指定．
例えば，"source/speaker\_1"が話者1のディレクトリの様な構造を持ち，" --source\_dir source/"と指定された場合には，"--speaker_dir_layer 1"を指定する.

### Another outputs
"--with\_speaker\_id"が指定された場合，--output\_dirの中に"speaker\_id.npz"という名前で話者を表すnpzファイルが作成される．

----
## unpack.py
collect2npz.pyで圧縮した特徴量ファイルをもとの各ファイルに戻す．
拡張子はもとのファイルの名前が利用される．
例えば，"mfcc.npz"を指定した場合，展開された各ファイルの拡張子は".mfcc"になる．

### Usage sample
```
python unpack.py \
  --source <SOURCE> \
  --output_dir <OUTPUT_DIR> \
  --keep_dir
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 対象ファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --output_dir | \[\*1\] | 出力ディレクトリ |
| --keep_dir | | ディレクトリ構造を保つ |

\[\*1\]:
--source_dirの親ディレクトリ

### Another outputs
"--with\_speaker\_id"が指定された場合，--output\_dirの中に"speaker\_id.npz"という名前で話者を表すnpzファイルが作成される．

----
## clean.py
extract.py / extract_world.pyで抽出した特徴量ファイルを削除する．

### Usage sample
```
python clean.py \
  --source_dir <SOURCE_DIR> \
  --remove_extensions <REMOVE_EXTENSIONS> \
```

### Arguments

#### Required arguments
None

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --source_dir | ./ | 対象ファイルのディレクトリ |
| --remove_extensions | \[\*1\] | 削除するファイル拡張子 |

\[\*1\]:
デフォルト値に以下の拡張子が指定されている．
* mfcc
* dmfcc
* ddmfcc
* mcep
* mspec
* logmspec
* spenv
* ap
* f0
* phn
* Ft_pnh
* wrd
* Ft_wrd
* npz

### Another outputs
None

----
## delta.py
微分を近似計算．

### Usage sample
```
python delta.py \
  --source <SOURCE> \
  --output <OUTPUT> \
  --size <SIZE>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 対象ファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --output | \[\*1\] | 出力ファイル |
| --size | 2 | 微分近似で用いる点数 |

\[\*1\]:
指定が無い場合は，入力ファイル名先頭に"delta\_"を付したファイルに出力．
例えば，"--source dir1/dir2/source.npz"が指定されたとき，出力ファイルは"dir1/dir2/delta\_source.npz"．

### Another outputs
None

----
## concatenate.py
幾つかの特徴量ファイルを結合．

### Usage sample
```
python concatenate.py \
  --sources <SOURCES> \
  --output <OUTPUT>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --sources | 対象ファイル (複数選択) |
| --output | 出力ファイル |

#### Optional arguments
None

### Another outputs
None

----
## filtering.py
移動平均をとる．

### Usage sample
```
python filtering.py \
  --source <SOURCE> \
  --output <OUTPUT> \
  --width <WIDTH> \
  --mode <MODE> \
  --triangle_grad <TRIANGLE_GRAD> \
  --triangle_bias <TRIANGLE_BIAS> \
  --gaussian_sigma <GAUSSIAN_SIGMA> \
  --scale <SCALE>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 対象ファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --output | \[\*1\] | 出力ファイル |
| --width | 5 | 窓サイズ |
| --mode | none | none / triangle / gaussian のうちどれか |
| --triangle_grad | 1.0 | 三角窓の傾き |
| --triangle_bias | 1.0 | 三角窓の切片 |
| --gaussian_sigma | 1.0 | ガウス窓の分散 |
| --scale | 1.0 | スケール (全体に掛ける) |

\[\*1\]:
指定が無い場合は，入力ファイル名先頭に"filt\_"を付したファイルに出力．
例えば，"--source dir1/dir2/source.npz"が指定されたとき，出力ファイルは"dir1/dir2/filt\_source.npz"．

### Another outputs
None

----
## separate_feature.py
次元を切り分ける (concatenateの逆)

### Usage sample
```
python separate_feature.py \
  --source <SOURCE> \
  --recipe <RECIPE>
  --single_prefix <SINGLE_PREFIX> \
  --single_suffix <SINGLE_SUFFIX> \
  --prefix <PREFIX> \
  --suffix <SUFFIX>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 対象ファイル |
| --recipe | 分割次元 \[\*1\] |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --show_dimension | | 入力ファイルの次元数を表示して終了 |
| --single_prefix | | 出力が1次元であった時の先頭の文字列 \[\*2\] |
| --single_suffix | (#b) | 出力が1次元であった時の末尾の文字列 \[\*2\] |
| --prefix | | 先頭の文字列 \[\*2\] |
| --suffix | (#b-#e) | 末尾の文字列 \[\*2\] |


\[\*1\]:
分割する次元数を順に指定する．
例えば，13次元の特徴量を5, 5, 3次元に分割したい際には，"--recipe 5 5 3"と指定する．
また，5, 5, 残りのように分割したい際は，"--recipe 5 5 0"のように0を用いることができる．
しかし，0は一度しか利用できない．

\[\*2\]:
\#bは対象次元のうちの先頭，\#eは対象次元のうちの末尾を表す．
例えば，13次元を"--recipe 7 6"のように分割した際には，初めの7次元の特徴量は\#b=0, \#e=6となり，残りの6次元の特徴量は\#b=7, \#e=12となる．

### Another outputs
None

----
## pickup.py
特徴量ファイルのうち特定のデータのみ抜き出す．

### Usage sample
```
python pickup.py \
  --source <SOURCE> \
  --output <OUTPUT>
  --regexp \
  --keyword <KEYWORD>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --source | 対象ファイル |
| --output | 出力ファイル |
| --keyword | キーワード |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --regexp | | --keywordを正規表現として扱う |

### Another outputs
None

----
## interactive_plot.py
特徴量ファイルのうち特定のデータのみ折れ線表示する．

### Usage sample
```
python interactive_plot.py \
  --target <TARGET>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --target | 対象ファイル |

#### Optional arguments
None

### Another outputs
None

----
## interactive_heatmap.py
特徴量ファイルのうち特定のデータのみヒートマップ表示する．

### Usage sample
```
python interactive_heatmap.py \
  --target <TARGET> \
  --cmap <CMAP>
```

### Arguments

#### Required arguments
| Argument | Help |
|----------|------|
| --target | 対象ファイル |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --cmap | jet | カラーマップ |

### Another outputs
None
