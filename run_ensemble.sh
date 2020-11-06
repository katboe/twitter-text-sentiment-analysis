#!/bin/bash


ENSEMBLE="Majority"
MODEL_FILE="modellist.txt"
CV=1

function fail {
	echo >&2 $1
	exit 1
}

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--ensemble)
    #size of data set (full/small)
    ENSEMBLE="$2"
    shift
    shift   
    ;;
    -m|--modelfile)
    #size of data set (full/small)
    MODEL_FILE="$2"
    shift
    shift   
    ;;
    -d|--debug)
    #size of data set (full/small)
    CV="$2"
    shift
    shift   
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

#run models if prediction files not present


#run ensemble method
if [[ ${ENSEMBLE} == 'Majority' ]]
then
	python3 utils/ensemble/majorityVoting.py --simple --model_file ${MODEL_FILE} --${CV}

elif [[ ${ENSEMBLE} == 'WeightedMajority' ]]
then
	python3 utils/ensemble/majorityVoting.py --weighted --model_file ${MODEL_FILE} --${CV}

elif [[ ${ENSEMBLE} == 'RandomForest' ]]
then
	python3 utils/ensemble/randomForest.py --model_file ${MODEL_FILE} --${CV}

elif [[ ${ENSEMBLE} == 'NN' ]]
then
	python3 utils/ensemble/neuralNetwork.py --model_file ${MODEL_FILE}
else
    echo "Specified ensemble method not implemented."
fi