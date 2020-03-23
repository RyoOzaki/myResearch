#!/bin/bash

#==============================================
# Volume normalization
python src/Volume_normalizer/normalize.py \
  --source_dir dataset/ \
  --max_amplitude 30000 \
  --replace

#==============================================
# extract mcep, phn, wrd, speaker_id by multispeaker_AIOI data
# output
#   feature/mfcc_all_speaker_20msec.npz (renamed from mfcc.npz)
#   feature/dmfcc_all_speaker_20msec.npz (renamed from dmfcc.npz)
#   feature/ddmfcc_all_speaker_20msec.npz (renamed from ddmfcc.npz)
#   feature/concat_mfcc_all_speaker_20msec.npz (concat mfcc, dmfcc, ddmfcc and rename)
#   feature/phn_all_speaker_20msec.npz
#   feature/wrd_all_speaker_20msec.npz
#   feature/Ft_phn_all_speaker_20msec.npz
#   feature/Ft_wrd_all_speaker_20msec.npz
#   feature/speaker.npz

python src/Extractor/extract.py \
  --source_dir dataset/ \
  --feature_type mfcc \
  --winlen 20E-3 \
  --winstep 20E-3 \
  --nfft 2048 \
  --label_format time \
  --phn_label_extension lab \
  --wrd_label_extension lab2 \
  --sil_label s

python src/Extractor/collect2npz.py \
  --source_dir dataset/ \
  --output_dir feature/ \
  --collect_extensions mfcc dmfcc ddmfcc phn wrd Ft_phn Ft_wrd \
  --with_speaker_id \
  --speaker_dir_layer 1

mv feature/mfcc.npz feature/mfcc_all_speaker_20msec.npz
mv feature/dmfcc.npz feature/dmfcc_all_speaker_20msec.npz
mv feature/ddmfcc.npz feature/ddmfcc_all_speaker_20msec.npz
mv feature/wrd.npz feature/wrd_all_speaker_20msec.npz
mv feature/phn.npz feature/phn_all_speaker_20msec.npz
mv feature/Ft_wrd.npz feature/Ft_wrd_all_speaker_20msec.npz
mv feature/Ft_phn.npz feature/Ft_phn_all_speaker_20msec.npz

python src/Extractor/concatenate.py \
  --sources \
    feature/mfcc_all_speaker_20msec.npz \
    feature/dmfcc_all_speaker_20msec.npz \
    feature/ddmfcc_all_speaker_20msec.npz \
  --output feature/concat_mfcc_all_speaker_20msec.npz

#==============================================
# extract mcep f0, ap by multispeaker_AIOI data
# output
#   feature/mcep_all_speaker_20msec.npz (renamed from mcep.npz)
#   feature/f0_all_speaker_20msec.npz (renamed from f0.npz)
#   feature/ap_all_speaker_20msec.npz (renamed from ap.npz)

python src/Extractor/extract_world.py \
  --source_dir dataset/ \
  --feature_type mcep f0 ap \
  --frame_period 20E-3 \
  --nfilt 36 \
  --fftsize 2048

python src/Extractor/collect2npz.py \
  --source_dir dataset/ \
  --output_dir feature/ \
  --collect_extensions mcep f0 ap

mv feature/mcep.npz feature/mcep_all_speaker_20msec.npz
mv feature/ap.npz feature/ap_all_speaker_20msec.npz
mv feature/f0.npz feature/f0_all_speaker_20msec.npz

#==============================================
# extract mcep by multispeaker_AIOI data
# output
#   feature/mcep_all_speaker_5msec.npz (renamed from mcep.npz)
#   feature/f0_all_speaker_5msec.npz (renamed from f0.npz)
#   feature/ap_all_speaker_5msec.npz (renamed from ap.npz)
python src/Extractor/extract_world.py \
  --source_dir dataset/ \
  --feature_type mcep f0 ap \
  --frame_period 5E-3 \
  --fftsize 2048 \
  --nfilt 36

python src/Extractor/collect2npz.py \
  --source_dir dataset/ \
  --output_dir feature/ \
  --collect_extensions mcep f0 ap

mv feature/mcep.npz feature/mcep_all_speaker_5msec.npz
mv feature/f0.npz feature/f0_all_speaker_5msec.npz
mv feature/ap.npz feature/ap_all_speaker_5msec.npz

#==============================================
# pickup each speaker
# output
#   feature/mfcc_speaker_H_20msec.npz (pickup by mfcc_all_speaker_20msec.npz)
#   feature/mfcc_speaker_K_20msec.npz (pickup by mfcc_all_speaker_20msec.npz)
#   feature/mfcc_speaker_M_20msec.npz (pickup by mfcc_all_speaker_20msec.npz)
#   feature/mfcc_speaker_N_20msec.npz (pickup by mfcc_all_speaker_20msec.npz)
speakers=("speaker_H" "speaker_K" "speaker_M" "speaker_N")

for spk in "${speakers[@]}"
do
  python src/Extractor/pickup.py \
    --source feature/mfcc_all_speaker_20msec.npz \
    --output feature/mfcc_${spk}_20msec.npz \
    --keyword ${spk}
done


#==============================================
# clean dataset dir
python src/Extractor/clean.py \
  --source_dir dataset/ \
  --remove_extensions mfcc dmfcc ddmfcc mcep f0 ap wrd Ft_wrd phn Ft_phn
