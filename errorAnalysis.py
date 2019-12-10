# -*- coding: utf-8 -*-
from codecs import open
import sys

#This script evaluates the systems on the SemEval 2018 task on Emoji Prediction.
#It takes the gold standard and system's output file as input and prints the results in terms of macro and micro average F-Scores (0-100).

def f1(precision,recall):
    return (2.0*precision*recall)/(precision+recall)


def main(path_goldstandard, path_outputfile):

    truth_dict={}
    output_dict_correct={}
    output_dict_attempted={}
    predicted = {}
    truth_file_lines=open(path_goldstandard,encoding='utf8').readlines()
    submission_file_lines=open(path_outputfile,encoding='utf8').readlines()
    if len(submission_file_lines)!=len(truth_file_lines): sys.exit('ERROR: Number of lines in gold and output files differ')
    for i in range(20):
        predicted[i] = [0 for _ in range(20)]
    for i in range(len(submission_file_lines)):
        line = submission_file_lines[i]
        emoji_code_gold = truth_file_lines[i]
        emoji_code_output=int(submission_file_lines[i].replace("\n",""))
        emoji_code_gold=int(truth_file_lines[i].replace("\n",""))
        predicted[emoji_code_gold][emoji_code_output] += 1
    
    for i in range(len(predicted)):
        total = sum(predicted[i])
        for j in range(len(predicted)):
            predicted[i][j] = str(round(predicted[i][j]/total*100,3)) + "%"
            if j < len(predicted) - 1:
                print(predicted[i][j], end = ", ")
            else:
                print(predicted[i][j])




if __name__ == '__main__':

    args = sys.argv[1:]

    if len(args) >= 2:

        path_goldstandard = args[0]
        path_outputfile = args[1]
        main(path_goldstandard, path_outputfile)
        
        
        
    else:
        sys.exit('''
            Requires:
            path_goldstandard -> Path of the gold standard
            path_outputfile -> Path of the system's outputfile
            ''')

