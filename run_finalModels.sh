#Compute all Models:
ENSEMBLE="NN"
MODEL_FILE="modellist.txt"

#LSTM Models
echo "LSTM_Models"
if ! [[ -f "results/predictions/LSTM_epochs2_ln32_full_enslp_word2vec.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e word2vec -c lstm -d 1
fi
if ! [[ -f "results/predictions/LSTM_epochs2_ln32_full_enslp_keras.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e keras -c lstm -d 1
fi


#Bidirectional LSTM Models
echo "BiLSTM_Models"
if ! [[ -f "results/predictions/BiLSTM_epochs2_ln32_full_enslp_word2vec.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e word2vec -c bilstm -d 1
fi
if ! [[ -f "results/predictions/BiLSTM_epochs2_ln32_full_enslp_keras.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e keras -c bilstm -d 1
fi


#CNN Models
echo "CNN_Models"
if ! [[ -f "results/predictions/CNN_epochs2_ln32_full_enslp_word2vec.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e word2vec -c cnn -d 1
fi
if ! [[ -f "results/predictions/CNN_epochs2_ln32_full_enslp_keras.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e keras -c cnn -d 1
fi

#CNN_LSTM Model
echo "CNN_LSTM_Models"
if ! [[ -f "results/predictions/CNN_LSTM_epochs2_ln32_full_enslp_word2vec.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e word2vec -c cnnlstm -d 1
fi
if ! [[ -f "results/predictions/CNN_LSTM_epochs2_ln32_full_enslp_keras.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e keras -c cnnlstm -d 1
fi

#GRU Models
echo "GRU_Models"
if ! [[ -f "results/predictions/GRU_epochs2_ln32_full_enslp_word2vec.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e word2vec -c gru -d 1
fi
if ! [[ -f "results/predictions/GRU_epochs2_ln32_full_enslp_keras.csv" ]]; then
	./run_classifier.sh -s full -p enslp -e keras -c gru -d 1
fi


#compute ensemble method
./run_ensemble.sh -e ${ENSEMBLE} --model_file ${MODEL_FILE}
