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
    truth_file_lines=open(path_goldstandard,encoding='utf8').readlines()
    submission_file_lines=open(path_outputfile,encoding='utf8').readlines()
    if len(submission_file_lines)!=len(truth_file_lines): sys.exit('ERROR: Number of lines in gold and output files differ')
    for i in range(len(submission_file_lines)):
        line=submission_file_lines[i]
        emoji_code_gold=truth_file_lines[i].replace("\n","")
        if emoji_code_gold not in truth_dict: truth_dict[emoji_code_gold]=1
        else: truth_dict[emoji_code_gold]+=1
        emoji_code_output=submission_file_lines[i].replace("\n","")
        if emoji_code_output==emoji_code_gold:
            if emoji_code_output not in output_dict_correct: output_dict_correct[emoji_code_gold]=1
            else: output_dict_correct[emoji_code_output]+=1
        if emoji_code_output not in output_dict_attempted: output_dict_attempted[emoji_code_output]=1
        else: output_dict_attempted[emoji_code_output]+=1
    precision_total=0
    recall_total=0
    num_emojis=len(truth_dict)
    attempted_total=0
    correct_total=0
    gold_occurrences_total=0
    f1_total=0
    for emoji_code in truth_dict:
        f1_score = 0
        gold_occurrences=truth_dict[emoji_code]
        if emoji_code in output_dict_attempted: attempted=output_dict_attempted[emoji_code]
        else: attempted=0
        if emoji_code in output_dict_correct: correct=output_dict_correct[emoji_code]
        else: correct=0
        if attempted!=0:
            precision=(correct*1.0)/attempted
            recall=(correct*1.0)/gold_occurrences
            print(precision, recall)
            precision_total += precision
            recall_total += recall
            if precision!=0.0 or recall!=0.0:
                f1_score = f1(precision, recall)
                f1_total+= f1_score

        print("emoji %s f1: "%(emoji_code) + str(round(f1_score*100,3)))
        attempted_total+=attempted
        correct_total+=correct
        gold_occurrences_total+=gold_occurrences
    macrof1=f1_total/(num_emojis*1.0)
    precision_total_micro=(correct_total*1.0)/attempted_total
    recall_total_micro=(correct_total*1.0)/gold_occurrences_total
    macro_precision = precision_total/(num_emojis*1.0)
    macro_recall = recall_total/(num_emojis*1.0)
    if precision_total_micro!=0.0 or recall_total_micro!=0.0: microf1=f1(precision_total_micro,recall_total_micro)
    else: microf1=0.0
    print ("Macro F-Score (official): "+str(round(macrof1*100,3)))
    print ("Macro Precision: "+str(round(macro_precision*100,3)))
    print ("Macro Recall: "+str(round(macro_recall*100,3)))
    print ("Accuracy: "+str(round(precision_total_micro*100,3)))
    print ("-----")
    print ("Micro F-Score: "+str(round(microf1*100,3)))
    print ("Micro Precision: "+str(round(precision_total_micro*100,3)))
    print ("Micro Recall: "+str(round(recall_total_micro*100,3)))


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

