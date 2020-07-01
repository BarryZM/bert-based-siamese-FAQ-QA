import torch
from transformers import *
from torch.utils.data import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time
import datetime
output_dir = './modelset/1p1n3epoch/'
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
class Sbert(nn.Module):
    def __init__(self):
        super(Sbert, self).__init__()
        self.bert= BertModel.from_pretrained(output_dir)
    def forward(self, in1,in1m):
        loss1, a = self.bert(in1, 
                             token_type_ids=None, 
                             attention_mask=in1m)
        input_mask_expanded1 = in1m.unsqueeze(-1).expand(loss1.size()).float()
        sum_embeddings1 = torch.sum(loss1 * input_mask_expanded1, 1)
        sum_mask1 = torch.clamp(input_mask_expanded1.sum(1), min=1e-9)
        output_vector1 = sum_embeddings1 / sum_mask1
        return output_vector1
tokenizer=BertTokenizer.from_pretrained(output_dir)
model=Sbert()
faqset=[]
with open("faq.txt","r") as f:
    while 1:
        lines=f.readline()
        if len(lines)==0:
            break
        faq=[]
        line=[e for e in lines.split()]
        faqanswer=f.readline()
        faq.append(int(line[0]))
        faqquestion=" ".join(line[1:])
        faq.append(faqquestion)
        faq.append(faqanswer)
        faqset.append(faq)
faq_ids=[]
faq_masks=[]
faqvector=[]
for i in range(125):
    encoded_dict1 = tokenizer.encode_plus(
                        faqset[i][1],                    
                        add_special_tokens = True, 
                        max_length = 100,          
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )   
    faqvector.append(model(encoded_dict1['input_ids'],encoded_dict1['attention_mask']))
inputline=input('How can I help you?(type bye to exit):')
while(len(inputline)!=0 and inputline!='bye'):
    encoded_dict2=tokenizer.encode_plus(
                        inputline,                    
                        add_special_tokens = True, 
                        max_length = 100,          
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )
    invector=model(encoded_dict2['input_ids'],encoded_dict2['attention_mask']) 
    mind=0
    mval=-1
    for i in range(125):
        output=torch.cosine_similarity(faqvector[i],invector)
        if output>mval:
            mind=i
            mval=output
    print(faqset[mind][2])
    inputline=input('How can I help you?(type bye to exit):')