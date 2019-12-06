# Extractor

## how to use (example)

```
python extract.py --source_dir ../multi_aioi --../multi_aioirate 16000 --feature_type all --label_format time --phn_label_extension lab --wrd_label_extension lab2
python collect2npz.py --source_dir ../multi_aioi --with_speaker_id --speaker_dir_layer 1
python separate_feature.py --source_file ../multi_aioi/mfcc.npz --recipe 1 12
python unpack.py --target ../multi_aioi/mfcc.npz --output_dir ../multi_aioi/ --keep_dir
python plot.py --target ../multi_aioi/mfcc.npz --save_dir ./figure
```

```
python extract.py --source_dir ../multi_aioi --feature_type mfcc --winfunc hamming --nfilt 36 --winlen 0.005 --nfft 2048 --winstep 0.005 --label_format time --phn_label_extension lab --wrd_label_extension lab2
python collect2npz.py --source_dir ../multi_aioi/ --collect_extensions mfcc phn wrd
python extract_world.py --source_dir ../multi_aioi/ --feature_type mcep --label_format time --phn_label_extension lab --wrd_label_extension lab2 --frame_period 5E-3 --nfilt 36
python collect2npz.py --source_dir ../multi_aioi/ --collect_extensions mcep phn wrd
```
```
python collect2npz.py --source_dir ../multi_aioi/ --output_dir result_world --collect_extensions mcep phn wrd
```


```
python extract_world.py --source_dir ../multi_aioi/speaker_K --feature_type mcep f0 --label_format time --phn_label_extension lab --wrd_label_extension lab2 --frame_period 0.005 --nfilt 36
python collect2npz.py --source_dir ../multi_aioi/speaker_K --output_dir ../multi_aioi/ --collect_extensions mcep f0 phn wrd
```
