#!/bin/bash

# Preprocessing experiment
# assumes there is the small training dataset in twitterData/ and creates files in data/

echo "Starting preprocessing experiment"

echo "None"
python3 utils/preprocess/preprocess.py --srcDir twitterData --small --none
python3 utils/preprocess_exp.py none

echo "Num"
python3 utils/preprocess/preprocess.py --srcDir twitterData --small --num
python3 utils/preprocess_exp.py num

echo "Pun"
python3 utils/preprocess/preprocess.py --srcDir twitterData --small --pun
python3 utils/preprocess_exp.py pun

echo "Rep"
python3 utils/preprocess/preprocess.py --srcDir twitterData --small --rep
python3 utils/preprocess_exp.py rep

echo "Sw"
python3 utils/preprocess/preprocess.py --srcDir twitterData --small --sw
python3 utils/preprocess_exp.py sw

echo "Stem"
python3 utils/preprocess/preprocess.py --srcDir twitterData --small --stem
python3 utils/preprocess_exp.py stem

echo "Lem"
python3 utils/preprocess/preprocess.py --srcDir twitterData --small --lem
python3 utils/preprocess_exp.py lem

echo "Emojis"
python3 utils/preprocess/preprocess.py --srcDir twitterData --small --emo
python3 utils/preprocess_exp.py emo

echo "Final"
python3 utils/preprocess/preprocess.py --srcDir twitterData --small --enslp
python3 utils/preprocess_exp.py enslp

