#!/bin/bash

EMBEDDING="keras"
SIZE="full"
CLASSIFIER="lstm"
CV="0"

function fail {
	echo >&2 $1
	exit 1
}

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -s|--size)
    SIZE="$2"
    shift
    shift
    ;;
    -p|--preprocess)
    PREPROCESS="$2"
    shift
    shift
    ;;
    -e|--embedding)
    EMBEDDING="$2"
    shift
    shift
    ;;
    -c|--classifier)
    CLASSIFIER="$2"
    shift
    shift
    ;;
     -d|--debug)
    CV="$2"
    shift
    shift
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

#run setup
bash run_setup.sh -s ${SIZE} -p ${PREPROCESS} -e ${EMBEDDING}


if [[ ${CLASSIFIER} == 'linear' ]]
then
    #run linear model (only for baseline)
    python3 utils/linearModel.py  --${SIZE} --${PREPROCESS} --${EMBEDDING} --${CV}|| fail "Could not train model"

else
    #run keras model (deep neural networks)
    python3 utils/model.py  --${SIZE} --${PREPROCESS} --${EMBEDDING} --${CLASSIFIER} --${CV}|| fail "Could not train model"
fi

echo "model trained"
