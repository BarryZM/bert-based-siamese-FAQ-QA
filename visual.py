import torch
from transformers import *
import torch.nn as nn
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
model=BertModel.from_pretrained('bert-base-uncased')
##########read file###########
faqset=[]
with open("faq.txt","r") as f:
    while 1:
        lines=f.readline()
        if len(lines)==0:
            break
        line=[e for e in lines.split()]
        faqquestion=" ".join(line[1:])
        faqanswer=f.readline()
        faqset.append([int(line[0]),faqquestion,faqanswer])
querydata=[]
with open("traindata.txt","r") as f:
    while 1:
        lines=f.readline()
        if len(lines)==0:
            break
        line=[e for e in lines.split()]
        query=" ".join(line[1:])
        querydata.append([int(line[0]),query])
##################encode data#################
faq_ids=[]
faq_masks=[]
for i in range(125):
    encoded_dict1 = tokenizer.encode_plus(
                        faqset[i][1],                    
                        add_special_tokens = True, 
                        max_length = 100,          
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )   
    faq_ids.append(encoded_dict1['input_ids'])
    faq_masks.append(encoded_dict1['attention_mask'])
faq_ids = torch.cat(faq_ids, dim=0)
faq_masks = torch.cat(faq_masks, dim=0)
'''
query_ids=[]
query_masks=[]
for i in range(750):
    encoded_dict1 = tokenizer.encode_plus(
                        querydata[i][1],                    
                        add_special_tokens = True, 
                        max_length = 100,          
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )   
    query_ids.append(encoded_dict1['input_ids'])
    query_masks.append(encoded_dict1['attention_mask'])
query_ids = torch.cat(query_ids, dim=0)
query_masks = torch.cat(query_masks, dim=0)
'''
loss, a = model(faq_ids, 
                 token_type_ids=None, 
                 attention_mask=faq_masks)
#############pooling############################
#avg#

input_mask_expanded = faq_masks.unsqueeze(-1).expand(loss.size()).float()
sum_embeddings = torch.sum(loss* input_mask_expanded, 1)
sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
output = sum_embeddings / sum_mask
'''
#[cls]token#
output=loss[:, 0, :]
#max#

input_mask_expanded = faq_masks.unsqueeze(-1).expand(loss.size()).float()
loss[input_mask_expanded == 0] = -1e9 
output = torch.max(loss, 1)[0] 
'''
###############show data##########################
output=output.tolist()
pca = decomposition.PCA(n_components=3)
pca.fit(output)
output = pca.transform(output)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in output:
    ax.scatter(i[0],i[1],i[2],s=10, c='b', marker='^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
