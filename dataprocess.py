traindata=[]
faqquestion=[]
faqanswer=[]
##########################################################
with open("stackExchange-FAQ.xml","r") as f:
    f.readline() #read rootflag
    for i in range(125):
        f.readline() #read qapairflag
        flag=f.readline()
        data=[]
        while flag=='<rephr>\n':
            d=f.readline()
            if d!='*\n':
                data.append(d)
            f.readline() #read <\rephr>
            flag=f.readline()
        traindata.append(data)
        query=f.readline()
        faqquestion.append(query) #read question flag
        f.readline() #read question flag
        flag=f.readline()
        aset=[]
        while flag=='<answer>\n':
            answer=f.readline()
            aset.append(answer)
            f.readline()
            flag=f.readline()
        faqanswer.append(aset)
    f.readline() #read rootflag
##############################################################
with open("traindata.txt","w") as f:
    for i in range(125):
        for e in traindata[i]:
            line=str(i)+' '+e
            f.writelines(line)
###############################################################
with open("faq.txt","w") as f:
    for i in range(125):
        line=str(i)+' '+faqquestion[i]+faqanswer[i][0]
        f.writelines(line)
        
'''
with open("faq1.txt","w") as f:
    for i in range(125):
        for e in faqanswer[i]:
            line=str(i)+' '+faqquestion[i]+e
            f.writelines(line)
'''