#!/bin/bash

# sh runner.sh -t datas/datas.npz -p datas/phn_all_speaker_20msec.npz -w datas/wrd_all_speaker_20msec.npz

label=sample_results
begin=1
end=20

while getopts t:p:w:l:b:e: OPT
do
  case $OPT in
    "t" ) train_data="${OPTARG}" ;;
    "p" ) phn_label="${OPTARG}" ;;
    "w" ) wrd_label="${OPTARG}" ;;
    "l" ) label="${OPTARG}" ;;
    "b" ) begin="${OPTARG}" ;;
    "e" ) end="${OPTARG}" ;;
  esac
done

mkdir -p ${label}

cp -r hypparams/ ${label}/
cp ${train_data} ${label}/
cp ${phn_label} ${label}/
cp ${wrd_label} ${label}/

mkdir -p results
mkdir -p parameters
mkdir -p summary_files

for i in `seq ${begin} ${end}`
do
  echo ${i}

  i_str=$( printf '%02d' $i )
  rm -rf results/
  rm -rf summary_files/
  rm -rf parameters/
  rm -rf figures/
  rm -rf models/
  rm -rf log.txt

  echo "#!/bin/bash" > continue.sh
  echo "sh src/NPB-DAA/runner.sh -t ${train_data} -p ${phn_label} -w ${wrd_label} -l ${label} -b ${i} -e ${end}" >> continue.sh

  python src/NPB-DAA/train.py --train_data ${train_data} | tee log.txt
  echo "summary starting..." >> log.txt
  python src/NPB-DAA/summary.py --phn_label ${phn_label} --wrd_label ${wrd_label} | tee -a log.txt
  echo "summary finished" >> log.txt

  mkdir -p ${label}/${i_str}/
  cp -r results/ ${label}/${i_str}/
  cp -r parameters/ ${label}/${i_str}/
  cp -r models/ ${label}/${i_str}/
  cp -r figures/ ${label}/${i_str}/
  cp -r summary_files/ ${label}/${i_str}/
  cp log.txt ${label}/${i_str}/

done

rm -rf continue.sh
rm -rf results/
rm -rf summary_files/
rm -rf parameters/
rm -rf figures/
rm -rf models/
rm -rf log.txt

echo "summary_summary starting..." >> log.txt
python src/NPB-DAA/summary_summary.py --result_dir ${label} | tee -a log.txt
cp -r summary_files ${label}
cp -r figures ${label}
echo "summary_summary finished" >> log.txt

rm -rf results/
rm -rf summary_files/
rm -rf parameters/
rm -rf figures/
rm -rf models/
rm -rf log.txt
