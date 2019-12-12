project report: 5525_report.pdf
project slides: 5525_emoji.pdf

Dataset:train, test, trial. Because the training set is too big so we only include the crawler

model code: Random.py, FFNN.py, BERT+LSTM+LEAVE_HEART.py, LSTM.ipynb, BERT+LSTM.ipynb
model result over test set: predictions1.txt, predictions2.txt, predictions3.txt, predictions4.txt, predictions5.txt
Random.py is the code of random selection
FFNN.py is the code of FFNN
LSTM.ipynb is the code of pytorch embedding + LSTM model
BERT+LSETM use BERT pretrained embedding + LSTM model
BERT+LSETM+LEAVE_HEART use model the same as above and use data that without red_heart 

analysis code: analysis.ipynb
analysis.ipynb do the error analysis over the prediction result. Change the path_outputfile in block 2 will show analysis of different models


useful link
project code:https://github.com/h11128/Multilingual-Emoji-Prediction
challenge website: https://competitions.codalab.org/competitions/17344#learn_the_details
evaluation_script:https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection/tree/master/tools/evaluation%20script
paper_related:https://www.aclweb.org/anthology/S18-1004/
challenge result sheet:https://docs.google.com/spreadsheets/d/1U2bALNkHStvdy-tMpKzu_y53jbwg3lCpTdOOXcR9N5Y/edit?usp=sharing

