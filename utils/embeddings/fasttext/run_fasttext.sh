#!/bin/bash

SIZE="full"
PREPROCESS="none"
DATA="data"
EMB="emb/embeddings"

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
    -d|--dataDir)
    DATA="$2"
    shift
    shift
     ;;
    -e|--embDir)
    EMB="$2"
    shift
    shift
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

# compute word embeddings
. utils/embeddings/fasttext/fasttext.config || fail "Fasttext Config File could not be read."

echo "fasttext config file read"


python3 utils/embeddings/fasttext/fasttext.py --${SIZE} --${PREPROCESS} --dataDir ${DATA} --embDir ${EMB}|| fail "Could not compute FastText Embedding"
echo "FastText Embedding computed"
