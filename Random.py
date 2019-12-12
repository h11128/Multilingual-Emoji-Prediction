import numpy as np 
import random

def random_pick(list,probabilities):
    x=random.uniform(0,1)
    cumulative_prob=0.0
    for item,item_prob in zip(list,probabilities):
        cumulative_prob+=item_prob
        if x<cumulative_prob:
            break
    return item

#Get frequency of each label
path='./tweet-train.txt.labels'
f=open(path,'r',encoding='utf-8')
# weight array 20*1
weight_array=np.zeros(20)
lines=f.readlines()
amount=len(lines)
for line in lines:
    index=int(line.rstrip('\n'))
    weight_array[index]+=1
probs=list(weight_array/amount)
f.close()

# predict on testset
test_path='./test/us_test.txt.labels'
output='./randomselection.txt.labels'
f_output=open(output,'w',encoding='utf-8')
f_test=open(test_path,'r',encoding='utf-8')
lines_test=f_test.readlines()
amount_test=len(lines_test)
labels=[i for i in range(0,20)]
predict_list=[]
for i in range(0,amount_test):
    label=random_pick(labels,probs)
    f_output.write(str(label)+'\n')
    predict_list.append(label)

#evaluation
correct=0
for i in range(0,amount_test):
    label=int(lines_test[i].rstrip('\n'))
    if label==predict_list[i]:
        correct+=1
print('accuracy on test is :'+str(correct/amount_test))
f_test.close()

