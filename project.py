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
import os
from model import *
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
#########set random seed#############
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
#########model################
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
model=Sbert()
########todevice#########################
device = torch.device("cpu")  # torch.device('cuda') if cuda  is available
#model.cuda() //use if cuda  is available
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
##########split data################
random.shuffle(querydata)
train_datas=querydata[:750]
valid_data=querydata[750:1000]
test_data=querydata[1000:]
##########enlarge training sample##########
negative_sum=1
positive_sum=negative_sum
train_data=[]
for i in range(750):
    for j in range(positive_sum):
        train_data.append([train_datas[i][0],train_datas[i][1],1.0])  
    randn=random.sample(list(range(0,train_datas[i][0]))+list(range(train_datas[i][0]+1,125)),k=negative_sum)
    for j in range(negative_sum):
        train_data.append([randn[j],train_datas[i][1],0.0])
#########store data###############
with open("train_set.txt","w") as f:
    for i in range(750*2*negative_sum):
        f.writelines(str(train_data[i]))
        f.writelines(faqset[train_data[i][0]][1])
        f.writelines("\n")
with open("valid_set.txt","w") as f:
    for i in range(250):
        f.writelines(str(valid_data[i]))
        f.writelines(faqset[valid_data[i][0]][1])
        f.writelines("\n")
with open("test_set.txt","w") as f:
    for i in range(250):
        f.writelines(str(test_data[i]))
        f.writelines(faqset[test_data[i][0]][1])
        f.writelines("\n")
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
input_ids1 = []
attention_masks1 = []
input_ids2 = []
attention_masks2 = []
labels=[]
for i in range(750*(negative_sum+positive_sum)):
    encoded_dict1 = tokenizer.encode_plus(
                        train_data[i][1],                    
                        add_special_tokens = True, 
                        max_length = 100,          
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )   
    input_ids1.append(encoded_dict1['input_ids'])
    attention_masks1.append(encoded_dict1['attention_mask'])

    input_ids2.append(faq_ids[train_data[i][0]])
    attention_masks2.append(faq_masks[train_data[i][0]])
    labels.append(torch.tensor([train_data[i][2]]))
vinput_ids1 = []
vattention_masks1 = []
vinput_ids2 = []
vattention_masks2 = []
vlabels=[]
for i in range(250):
    encoded_dict1 = tokenizer.encode_plus(
                        valid_data[i][1],                    
                        add_special_tokens = True, 
                        max_length = 100,          
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )   
    vinput_ids1.append(encoded_dict1['input_ids'])
    vattention_masks1.append(encoded_dict1['attention_mask'])

    vinput_ids2.append(faq_ids[valid_data[i][0]])
    vattention_masks2.append(faq_masks[valid_data[i][0]])
    vlabels.append(torch.tensor([1.0]))

input_ids1 = torch.cat(input_ids1, dim=0)
attention_masks1 = torch.cat(attention_masks1, dim=0)
input_ids2 = torch.cat(input_ids2, dim=0)
attention_masks2 = torch.cat(attention_masks2, dim=0)
labels = torch.cat(labels, dim=0)
vinput_ids1 = torch.cat(vinput_ids1, dim=0)
vattention_masks1 = torch.cat(vattention_masks1, dim=0)
vinput_ids2 = torch.cat(vinput_ids2, dim=0)
vattention_masks2 = torch.cat(vattention_masks2, dim=0)
vlabels = torch.cat(vlabels, dim=0)

batch_size = 32

train_dataset = TensorDataset(input_ids1, attention_masks1,input_ids2,attention_masks2,labels)
train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size 
        )

valid_dataset = TensorDataset(vinput_ids1, vattention_masks1,vinput_ids2,vattention_masks2,vlabels)
valid_dataloader = DataLoader(
            valid_dataset,  
            sampler = SequentialSampler(valid_dataset), 
            batch_size = batch_size 
        )

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )

epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

total_t0 = time.time()

#########training#####################
for epoch_i in range(0, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step+1, len(train_dataloader), elapsed))

        b_input_ids1 = batch[0].to(device)
        b_input_mask1 = batch[1].to(device)
        b_input_ids2 = batch[2].to(device)
        b_input_mask2 = batch[3].to(device)
        b_labels = batch[4].to(device)

        model.zero_grad()        

        loss=model(b_input_ids1,b_input_mask1,b_input_ids2,b_input_mask2,b_labels)
        total_train_loss += loss.item()
        print("total loss:",total_train_loss,"\naverage loss:",total_train_loss/(step+1),"\n--------------------------------------------------------")
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    print("")
    print("Running Validation...")
    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0

    for step, batch in enumerate(valid_dataloader):

        vb_input_ids1 = batch[0].to(device)
        vb_input_mask1 = batch[1].to(device)
        vb_input_ids2 = batch[2].to(device)
        vb_input_mask2 = batch[3].to(device)
        vb_labels = batch[4].to(device)
        
        with torch.no_grad():        
            vloss=model(vb_input_ids1,vb_input_mask1,vb_input_ids2,vb_input_mask2,vb_labels)
            total_eval_loss += vloss.item()
            print("total loss:",total_eval_loss,"\naverage loss:",total_eval_loss/(step+1),"\n---------------------------------------------")
        
    avg_val_loss = total_eval_loss / len(valid_dataloader)
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

output_dir = './model_save/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)
model_to_save = model.bert.module if hasattr(model, 'module') else model.bert  
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

#########################################################


model.eval()

test_set=[]
for i in range(250):
    input_ids1t = []
    attention_masks1t = []
    for j in range(125):
        encoded_dict1 = tokenizer.encode_plus(
                            test_data[i][1],                    
                            add_special_tokens = True, 
                            max_length = 100,          
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',     
                        )   
        input_ids1t.append(encoded_dict1['input_ids'])
        attention_masks1t.append(encoded_dict1['attention_mask'])
    test_set.append([input_ids1t,attention_masks1t])
t0=time.time()
mapsum=0
accuracysum=0
for i in range(250):
    print(format_time(time.time()-t0))
    ansind=test_data[i][0]
    test_set[i][0][ansind]=test_set[i][0][ansind].to(device)
    test_set[i][1][ansind]=test_set[i][1][ansind].to(device)
    faq_ids[ansind]=faq_ids[ansind].to(device)
    faq_masks[ansind]=faq_masks[ansind].to(device)
    tlabels=torch.tensor([[1]]).to(device)
    standardscore=model(test_set[i][0][ansind],test_set[i][1][ansind],faq_ids[ansind],faq_masks[ansind],tlabels)
    rank=1
    for j in range(125):
        test_set[i][0][j]=test_set[i][0][j].to(device)
        test_set[i][1][j]=test_set[i][1][j].to(device)
        faq_ids[j]=faq_ids[j].to(device)
        faq_masks[j]=faq_masks[j].to(device)
        tlabels=torch.tensor([1]).to(device)
        output=model(test_set[i][0][j],test_set[i][1][j],faq_ids[j],faq_masks[j],tlabels)
        if output<standardscore:
            rank+=1
        print('q\n')
    mapsum+=1/rank
    if rank==1:
        accuracysum+=1
    print(mapsum/(i+1),accuracysum/(i+1))

print('MAP:',mapsum/250,'\nACCURACY:',accuracysum/250)    