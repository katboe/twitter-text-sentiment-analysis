#!/bin/bash

SIZE="full"
PREPROCESS="none"
DATA="data"
EMB="data/embeddings"

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
. utils/embeddings/word2vec/word2vec.config || fail "Word2Vec Config File could not be read."

echo "Word2Vec config file read"

#
python3 utils/embeddings/word2vec/word2vec.py --${SIZE} --${PREPROCESS} --dataDir ${DATA} --embDir ${EMB}|| fail "Could not compute Word2Vec Embedding"
echo "Word2Vec Embedding computed"
