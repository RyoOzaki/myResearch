#!/bin/bash

# sh watchdog_runner.sh -t datas/datas.npz -p datas/phn.npz -w datas/wrd.npz

killpstree(){
  kill -SIGSTOP $1
  children=`ps --ppid $1 --no-heading | awk '{ print $1 }'`
  for child in $children
  do
      killpstree $child
  done
  kill -SIGINT $1
}

INTERVAL=300 # sec
TARGET=log.txt # target file of watch
CONTINUE=false # flag of run runner.sh or continue.sh
WATCHDOG_LOG=watchdog_log.txt

label=sample_results
begin=1
end=20

while getopts ct:p:w:l:b:e: OPT
do
  case $OPT in
    "c" ) CONTINUE=true ;;
    "t" ) train_data="${OPTARG}" ;;
    "p" ) phn_label="${OPTARG}" ;;
    "w" ) wrd_label="${OPTARG}" ;;
    "l" ) label="${OPTARG}" ;;
    "b" ) begin="${OPTARG}" ;;
    "e" ) end="${OPTARG}" ;;
  esac
done

echo "start-up watchdog" >> ${WATCHDOG_LOG}
echo "--INTERVAL=${INTERVAL}" >> ${WATCHDOG_LOG}
echo "--TARGET=${TARGET}" >> ${WATCHDOG_LOG}
if "${CONTINUE}" ; then
  echo "--CONTINUE mode" >> ${WATCHDOG_LOG}
else
  echo "--label=${label}" >> ${WATCHDOG_LOG}
  echo "--begin=${begin}" >> ${WATCHDOG_LOG}
  echo "--end=${end}" >> ${WATCHDOG_LOG}
fi

touch ${TARGET}

last=`ls --full-time ${TARGET} | awk '{print $6"-"$7}'`

echo -n "start process..." >> ${WATCHDOG_LOG}
if "${CONTINUE}" ; then
  sh continue.sh &
else
  sh runner.sh -t ${train_data} -p ${phn_label} -w ${wrd_label} -l ${label} -b ${begin} -e ${end} &
fi
PID=$!
echo "done!" >> ${WATCHDOG_LOG}
echo "PID=${PID}" >> ${WATCHDOG_LOG}

trap 'echo "shutdown watchdog" >> ${WATCHDOG_LOG}; killpstree ${PID}; exit 1' 1 2 3 15

sleep ${INTERVAL}

while [ `ps -a | grep "${PID}" -o` ] ; do

  current=`ls --full-time ${TARGET} | awk '{print $6"-"$7}'`

  if [ ${last} != ${current} ] ; then
    last=$current
  else
    echo -n "shutdown process ${PID}..." >> ${WATCHDOG_LOG}
    killpstree ${PID}
    echo "done!" >> ${WATCHDOG_LOG}
    echo -n "rebooting process..." >> ${WATCHDOG_LOG}
    sh continue.sh &
    PID=$!
    echo "done!" >> ${WATCHDOG_LOG}
    echo "PID=${PID}" >> ${WATCHDOG_LOG}
  fi

  sleep ${INTERVAL}
done

echo "all process finished!" >> ${WATCHDOG_LOG}

exit 0
