#!/bin/bash

#This scripts sets up framework s.t. classifier can be trained on twitterData.
#It preprocesses the twitterData and trains the embedding if specified

SIZE="full"
PREPROCESS="none"
EMBEDDING="keras"
DIM=200

function fail {
	echo >&2 $1
	exit 1
}

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -s|--size)
    #size of data set (full/small)
    SIZE="$2"
    shift
    shift
    ;;
    -p|--preprocess)
    #type of preprocessing (none, num, enrs, enslp)
    PREPROCESS="$2"
    shift
    shift
    ;;
    -e|--embedding)
    #type of embedding (keras, word2vec, fasttext)
    EMBEDDING="$2"
    shift
    shift
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

. structure.config || fail "Structure config file could not be read."

echo "preprocess data"
# preprocess datasets and split files for cross validation
if ! [[ -f "data/train_${SIZE}_${PREPROCESS}.txt" ]]; then
	python3 utils/preprocess/preprocess.py --srcDir ${TWITTERDATA} --destDir ${DATA} --${SIZE} --${PREPROCESS}|| fail "Datasets could not be preprocessed."
fi
echo "data preprocessed"

if  ! [[ -f "data/cv_train_${SIZE}_${PREPROCESS}_1.txt" ]]; then
    python3 utils/preprocess/split.py --dataDir ${DATA} --${SIZE} --${PREPROCESS}|| fail "Cross validation split could not be preprocessed."
   
fi

 echo "data split into cross validation files"

# compute word embedding (optional)
if [[ ${EMBEDDING} != 'keras' ]]; then
    . utils/embeddings/${EMBEDDING}/${EMBEDDING}.config || fail "Embedding config file could not be read."

    echo "embedding config file read"

    #check if embeddings is already present (check size and preprocess)
    if ! [[ -f "data/embeddings/${EMBEDDING}_${SIZE}_${PREPROCESS}_${DIM}.${FORMAT}" ]] ; then
            bash utils/embeddings/${EMBEDDING}/run_${EMBEDDING}.sh -s ${SIZE} -p ${PREPROCESS} --embDir ${EMB} --dataDir ${DATA}|| fail "Could not compute embedding"
    fi

    echo "embedding computed"
fi