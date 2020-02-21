# README

----
## baseline_vocoder.py
ベースラインを用いて音声合成を行う．

### Usage sample
```
python baseline_vocoder.py \
  --samplerate <SAMPLERATE> \
  --fftsize <FFTSIZE> \
  --frame_period <FRAME_PERIOD> \
  --letter_num <LETTER_NUM> \
  --sentences_file <SENTENCES_FILE> \
  --letter_stateseq <LETTER_STATESEQ> \
  --ap <AP> \
  --f0 <F0> \
  --mcep <MCEP> \
  --parameter <PARAMETER> \
  --speaker_id <SPEAKER_ID> \
  --target_speaker <TARGET_SPEAKER> \
  --output_prefix <OUTPUT_PREFIX> \
  --sentences <SENTENCES_1> \
  --sentences <SENTENCES_2> \
  --sentences <SENTENCES_3> \
  --size <SIZE> \
  --mode <MODE> \
  --flat_f0 \
  --flat_ap \
  --LM <LANGUAGE_MODEL> \
  --LSTM_model <LSTM_MODEL> \
  --unique
```

### Arguments
#### Required arguments
| Argument | Help |
|----------|------|
| --target_speaker | 合成先の話者 |
| --sentences_file | センテンスファイル (made by LSTMLM/convert.py) |
| --speaker_id | 話者ファイル |
| --ap | 非周期性指標ファイル |
| --f0 | F0ファイル |
| --mcep | メルケプストラムファイル |
| --letter_num | 音素数 |
| --parameter | NPB-DAAのパラメータファイル |
| --letter_stateseq | NPB-DAAによる音素分節化結果ファイル |
| --output_prefix | 出力先 |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --samplerate | 48000 | サンプリング周波数 |
| --fftsize | 1024 | FFTサイズ |
| --frame_period | 0.005 | 窓・シフト長 (sec) |
| --sentences | | 合成する単語列 (複数指定可能) \[\*1\] |
| --mode | ML | 合成モード (ML: 最尤列, RND: 乱数列) |
| --flat_f0 | | F0の合成時に話者の発話全体の平均値を用いる \[\*2\] |
| --flat_ap | | apの合成時に話者の発話全体の平均値を用いる \[\*2\] |
| --LM | | 使用する言語モデル (LSTM / Bigram / Unigram) |
| --LSTM_model | | "--LM LSTM"を指定した際に使用するLSTMのモデルファイル |
| --size | 1 | "--LM"を指定した際に生成する文章数 \[\*1\] |
| --unique | | 言語モデルを用いた生成時に重複を許さない |

\[\*1\]:
"--sentences"が指定されている場合には，"--LM"オプションで指定された言語モデルを使用しない．

\[\*2\]:
各flatが指定されると，"--mode ML"よりもさらに平坦な合成音声になる．
具体的には，flatを指定しない場合は，NPB-DAAの分節化結果をもとに各音素の平均値・標準偏差を算出し利用する．
flatを指定した場合は，出力先話者の平均値を利用する．

### Another outputs
None

----
## dsae_vocoder.py
DSAEを用いて音声合成を行う．

### Usage sample
```
python dsae_vocoder.py \
  --samplerate <SAMPLERATE> \
  --fftsize <FFTSIZE> \
  --frame_period <FRAME_PERIOD> \
  --letter_num <LETTER_NUM> \
  --sentences_file <SENTENCES_FILE> \
  --letter_stateseq <LETTER_STATESEQ> \
  --ap <AP> \
  --f0 <F0> \
  --dsae_param <DSAE_PARAM> \
  --mcep_norm_param <MCEP_NORM_MIN> <MCEP_NORM_MAX> \
  --parameter <PARAMETER> \
  --speaker_id <SPEAKER_ID> \
  --target_speaker <TARGET_SPEAKER> \
  --output_prefix <OUTPUT_PREFIX> \
  --sentences <SENTENCES_1> \
  --sentences <SENTENCES_2> \
  --sentences <SENTENCES_3> \
  --size <SIZE> \
  --mode <MODE> \
  --flat_f0 \
  --flat_ap \
  --LM <LANGUAGE_MODEL> \
  --LSTM_model <LSTM_MODEL> \
  --unique
```

### Arguments
#### Required arguments
| Argument | Help |
|----------|------|
| --target_speaker | 合成先の話者 |
| --sentences_file | センテンスファイル (made by LSTMLM/convert.py) |
| --speaker_id | 話者ファイル |
| --ap | 非周期性指標ファイル |
| --f0 | F0ファイル |
| --dsae_param | DSAEのパラメータファイル |
| --mcep_norm_param | メルケプストラムの最小値・最大値ファイル |
| --letter_num | 音素数 |
| --parameter | NPB-DAAのパラメータファイル |
| --letter_stateseq | NPB-DAAによる音素分節化結果ファイル |
| --output_prefix | 出力先 |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --samplerate | 48000 | サンプリング周波数 |
| --fftsize | 1024 | FFTサイズ |
| --frame_period | 0.005 | 窓・シフト長 (sec) |
| --sentences | | 合成する単語列 (複数指定可能) \[\*1\] |
| --mode | ML | 合成モード (ML: 最尤列, RND: 乱数列) |
| --flat_f0 | | F0の合成時に話者の発話全体の平均値を用いる \[\*2\] |
| --flat_ap | | apの合成時に話者の発話全体の平均値を用いる \[\*2\] |
| --LM | | 使用する言語モデル (LSTM / Bigram / Unigram) |
| --LSTM_model | | "--LM LSTM"を指定した際に使用するLSTMのモデルファイル |
| --size | 1 | "--LM"を指定した際に生成する文章数 \[\*1\] |
| --unique | | 言語モデルを用いた生成時に重複を許さない |

\[\*1\]:
"--sentences"が指定されている場合には，"--LM"オプションで指定された言語モデルを使用しない．

\[\*2\]:
各flatが指定されると，"--mode ML"よりもさらに平坦な合成音声になる．
具体的には，flatを指定しない場合は，NPB-DAAの分節化結果をもとに各音素の平均値・標準偏差を算出し利用する．
flatを指定した場合は，出力先話者の平均値を利用する．

### Another outputs
None

----
## dsaepbhl_vocoder.py
DSAE-PBHLを用いて音声合成を行う．

### Usage sample
```
python dsaepbhl_vocoder.py \
  --samplerate <SAMPLERATE> \
  --fftsize <FFTSIZE> \
  --frame_period <FRAME_PERIOD> \
  --letter_num <LETTER_NUM> \
  --sentences_file <SENTENCES_FILE> \
  --letter_stateseq <LETTER_STATESEQ> \
  --ap <AP> \
  --f0 <F0> \
  --dsae_param <DSAE_PARAM> \
  --pb_param <PB_PARAM> \
  --mcep_norm_param <MCEP_NORM_MIN> <MCEP_NORM_MAX> \
  --parameter <PARAMETER> \
  --speaker_id <SPEAKER_ID> \
  --target_speaker <TARGET_SPEAKER> \
  --output_prefix <OUTPUT_PREFIX> \
  --sentences <SENTENCES_1> \
  --sentences <SENTENCES_2> \
  --sentences <SENTENCES_3> \
  --size <SIZE> \
  --mode <MODE> \
  --flat_f0 \
  --flat_ap \
  --LM <LANGUAGE_MODEL> \
  --LSTM_model <LSTM_MODEL> \
  --unique
```

### Arguments
#### Required arguments
| Argument | Help |
|----------|------|
| --target_speaker | 合成先の話者 |
| --sentences_file | センテンスファイル (made by LSTMLM/convert.py) |
| --speaker_id | 話者ファイル |
| --ap | 非周期性指標ファイル |
| --f0 | F0ファイル |
| --dsae_param | DSAEのパラメータファイル |
| --pb_param | DSAE-PBHLにおけるパラメトリックバイアスの平均値ファイル |
| --mcep_norm_param | メルケプストラムの最小値・最大値ファイル |
| --letter_num | 音素数 |
| --parameter | NPB-DAAのパラメータファイル |
| --letter_stateseq | NPB-DAAによる音素分節化結果ファイル |
| --output_prefix | 出力先 |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --samplerate | 48000 | サンプリング周波数 |
| --fftsize | 1024 | FFTサイズ |
| --frame_period | 0.005 | 窓・シフト長 (sec) |
| --sentences | | 合成する単語列 (複数指定可能) \[\*1\] |
| --mode | ML | 合成モード (ML: 最尤列, RND: 乱数列) |
| --flat_f0 | | F0の合成時に話者の発話全体の平均値を用いる \[\*2\] |
| --flat_ap | | apの合成時に話者の発話全体の平均値を用いる \[\*2\] |
| --LM | | 使用する言語モデル (LSTM / Bigram / Unigram) |
| --LSTM_model | | "--LM LSTM"を指定した際に使用するLSTMのモデルファイル |
| --size | 1 | "--LM"を指定した際に生成する文章数 \[\*1\] |
| --unique | | 言語モデルを用いた生成時に重複を許さない |

\[\*1\]:
"--sentences"が指定されている場合には，"--LM"オプションで指定された言語モデルを使用しない．

\[\*2\]:
各flatが指定されると，"--mode ML"よりもさらに平坦な合成音声になる．
具体的には，flatを指定しない場合は，NPB-DAAの分節化結果をもとに各音素の平均値・標準偏差を算出し利用する．
flatを指定した場合は，出力先話者の平均値を利用する．

### Another outputs
None

----
## stargan_vocoder.py
StarGAN-VCを用いて音声合成を行う．

### Usage sample
```
python stargan_vocoder.py \
  --samplerate <SAMPLERATE> \
  --fftsize <FFTSIZE> \
  --frame_period <FRAME_PERIOD> \
  --letter_num <LETTER_NUM> \
  --sentences_file <SENTENCES_FILE> \
  --letter_stateseq <LETTER_STATESEQ> \
  --ap <AP> \
  --f0 <F0> \
  --stargan_generator <STARGAN_GENERATOR> \
  --mcep_norm_param <MCEP_NORM_MIN> <MCEP_NORM_MAX> \
  --parameter <PARAMETER> \
  --speaker_id <SPEAKER_ID> \
  --target_speaker <TARGET_SPEAKER> \
  --output_prefix <OUTPUT_PREFIX> \
  --sentences <SENTENCES_1> \
  --sentences <SENTENCES_2> \
  --sentences <SENTENCES_3> \
  --size <SIZE> \
  --mode <MODE> \
  --flat_f0 \
  --flat_ap \
  --LM <LANGUAGE_MODEL> \
  --LSTM_model <LSTM_MODEL> \
  --unique
```

### Arguments
#### Required arguments
| Argument | Help |
|----------|------|
| --target_speaker | 合成先の話者 |
| --sentences_file | センテンスファイル (made by LSTMLM/convert.py) |
| --speaker_id | 話者ファイル |
| --ap | 非周期性指標ファイル |
| --f0 | F0ファイル |
| --stargan_generator | StarGAN-VCのGeneratorファイル |
| --mcep_norm_param | メルケプストラムの平均値・標準偏差ファイル |
| --letter_num | 音素数 |
| --parameter | NPB-DAAのパラメータファイル |
| --letter_stateseq | NPB-DAAによる音素分節化結果ファイル |
| --output_prefix | 出力先 |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --samplerate | 48000 | サンプリング周波数 |
| --fftsize | 1024 | FFTサイズ |
| --frame_period | 0.005 | 窓・シフト長 (sec) |
| --sentences | | 合成する単語列 (複数指定可能) \[\*1\] |
| --mode | ML | 合成モード (ML: 最尤列, RND: 乱数列) |
| --flat_f0 | | F0の合成時に話者の発話全体の平均値を用いる \[\*2\] |
| --flat_ap | | apの合成時に話者の発話全体の平均値を用いる \[\*2\] |
| --LM | | 使用する言語モデル (LSTM / Bigram / Unigram) |
| --LSTM_model | | "--LM LSTM"を指定した際に使用するLSTMのモデルファイル |
| --size | 1 | "--LM"を指定した際に生成する文章数 \[\*1\] |
| --unique | | 言語モデルを用いた生成時に重複を許さない |
| --reinput | | 指定時に合成したメルケプストラムを再度Generatorに通す |

\[\*1\]:
"--sentences"が指定されている場合には，"--LM"オプションで指定された言語モデルを使用しない．

\[\*2\]:
各flatが指定されると，"--mode ML"よりもさらに平坦な合成音声になる．
具体的には，flatを指定しない場合は，NPB-DAAの分節化結果をもとに各音素の平均値・標準偏差を算出し利用する．
flatを指定した場合は，出力先話者の平均値を利用する．

### Another outputs
None

----
## topline_vocoder.py
トップラインを用いて音声合成を行う．

### Usage sample
```
python topline_vocoder.py \
  --samplerate <SAMPLERATE> \
  --fftsize <FFTSIZE> \
  --frame_period <FRAME_PERIOD> \
  --phn <PHN> \
  --ap <AP> \
  --f0 <F0> \
  --mcep <MCEP> \
  --speaker_id <SPEAKER_ID> \
  --target_speaker <TARGET_SPEAKER> \
  --output_prefix <OUTPUT_PREFIX> \
  --key_of_pickuped_sentences <SENTENCES> \
  --mode <MODE> \
  --flat_f0 \
  --flat_ap
```

### Arguments
#### Required arguments
| Argument | Help |
|----------|------|
| --target_speaker | 合成先の話者 |
| --speaker_id | 話者ファイル |
| --ap | 非周期性指標ファイル |
| --f0 | F0ファイル |
| --mcep | メルケプストラムファイル |
| --key_of_pickuped_sentences | 合成する音声 (複数指定可) |
| --output_prefix | 出力先 |

#### Optional arguments
| Argument | Default | Help |
|----------|---------|------|
| --samplerate | 48000 | サンプリング周波数 |
| --fftsize | 1024 | FFTサイズ |
| --frame_period | 0.005 | 窓・シフト長 (sec) |
| --mode | ML | 合成モード (ML: 最尤列, RND: 乱数列) |
| --flat_f0 | | F0の合成時に話者の発話全体の平均値を用いる \[\*1\] |
| --flat_ap | | apの合成時に話者の発話全体の平均値を用いる \[\*1\] |

\[\*1\]:
各flatが指定されると，"--mode ML"よりもさらに平坦な合成音声になる．
具体的には，flatを指定しない場合は，NPB-DAAの分節化結果をもとに各音素の平均値・標準偏差を算出し利用する．
flatを指定した場合は，出力先話者の平均値を利用する．

### Another outputs
None
