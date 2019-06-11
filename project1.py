import pytesseract
import pandas as pd
import re
import os
import numpy as np
from nltk.corpus import stopwords
from PIL import Image,ImageEnhance
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import accuracy_score, classification_report
#-------------------------------------------------------------------EndOfimports-----------------------------------------------------------------------------
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
#----------------------------------------------------------HandlingSupresserFutureWarnings-------------------------------------------------------------------
# def main():
#     i=1
#     for filename in os.listdir(r'C:\\Users\\Vishal.Chauhan\\Desktop\\tessert'):
#         dst="test"+str(i)+".jpg"
#         src=r'C:\\Users\\Vishal.Chauhan\\Desktop\\tessert'+filename
#         dst=r'C:\\Users\\Vishal.Chauhan\\Desktop\\tessert'+dst
#         os.rename(src,dst)
#         i+=1
# if __name__=='__main__':
#     main()
for i in range(1,5):
    Extension=".jpg"
    FILE="\\"+"test"+str(i)
    EnhancedExtension="_enhanced"
    factor=2.65
    tesseractpath=r'C:\\Users\\Vishal.Chauhan\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'
    directory=r'C:\\Users\\Vishal.Chauhan\\Desktop\\tessert'
    testCsv='\dataset.csv'
    trainCsv='\Structure.csv'
    #-------------------------------------------------------------------EndOfFileInputs--------------------------------------------------------------------------
    pytesseract.pytesseract.tesseract_cmd=tesseractpath                                                                                             #setting path
    FileToBeEnhanced=Image.open(directory+FILE+Extension)                                                                                         #IMAGE LOCATION
    enhancer = ImageEnhance.Sharpness(FileToBeEnhanced)
    # enhancer = ImageEnhance.Color(FileToBeEnhanced)
    enhanced_im = enhancer.enhance(factor)
    enhanced_im.save(directory+FILE+EnhancedExtension+Extension)
    t=Image.open(directory+FILE+EnhancedExtension+Extension)                                                                                   #OpenEnhancedImage
    text=pytesseract.image_to_string(t,lang="eng")                                                                                              #ExtractionOfText
    print(text)
    text1=re.sub(r'[^a-zA-Z0-9#/_\s]+', '',text)                                                                                     #Removing special characters
    token=word_tokenize(text1)                                                                                                                      #tokenisation
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in token if not w in stop_words]                                                                      #stopwords   - a ,the , is
    filtered_sentence = []
    for w in token:
        if w not in stop_words:
            filtered_sentence.append(w)
    print(filtered_sentence)                                                                                                                    #stopwords output
    #words = [word for word in filtered_sentence if word.isalpha()]                                                                                #For Words Only
    numbers = [word for word in filtered_sentence if word.isnumeric()]                                                                          #For Numbers Only
    res = "".join(filter(lambda x: not x.isdigit(), filtered_sentence))
    print(str(res))
    #print(words)
    print(numbers)
    amount=numbers[-1]                                                                                                                         #To Extract Amount
    valueFetched=int(amount)/100
    print(valueFetched)
    #-----------------------------------------------------------EnterValueToDataFrame----------------------------------------------------------------------------
    type=[0];invoice_number=[];due_date=[];description=[];quantity=[];unit_price=[];bill_to=[];total=[];weight=[];cust_ref=[];retainage=[];ship_from=[];ship_to=[];ship_date=[];customer_id=[];pickup_date=[]
    #my_list_string=''.join(words)
    fdist=FreqDist()
    inv=re.findall(r"\BINVOICE#|\BINVOICENo|\BINVOICENO|\BINVOICENUMBER",str(res))
    due=re.findall(r"\BDUEDATE|\BD..D..e|\BDATED",str(res))#my_list_string
    des=re.findall(r"\BDes......on",str(res))
    qty=re.findall(r"\BQty|\BQ.....TY|\B.ty|\Barty",str(res))
    rate=re.findall(r"\BRATE",str(res))
    billed_to=re.findall(r"\BBILLTO|\BBillto|\BBiTo|\BB...to",str(res))
    total_value=re.findall(r"\BTotal|\BTOTAL",str(res))
    wt=re.findall(r"\BW....t",str(res))
    cst_ref=re.findall(r"\BRefNo|\BREFNO|\BR..N.|\BREFERENCENUMBER|\BREF#",str(res))
    ret=re.findall(r"\BRETAINAGE|\BRET....GE",str(res))
    shipped_from=re.findall(r"\BFROM",str(res))
    shipped_to=re.findall(r"\BTO",str(res))
    shipped_date=re.findall(r"\BTERMS",str(res))
    cst_id=re.findall(r"\BCustRef",str(res))
    pickedup_date=re.findall(r"\BDate|\BDATE|\BPICKUPDATE|\BTOWING DATE",str(res))
    for word_inv in inv:                                                                                                                       #ForInvoiceNumber
        fdist[word_inv.lower()]+=1
    for word_due in due:                                                                                                                             #ForDueDate
        fdist[word_due.lower()]+=1
    for word_des in des:                                                                                                                         #ForDescription
        fdist[word_des.lower()]+=1
    for word_qty in qty:                                                                                                                            #ForQuantity
        fdist[word_qty.lower()]+=1
    for word_rate in rate:                                                                                                                              #ForRate
        fdist[word_rate.lower()]+=1
    for word_billedto in billed_to:                                                                                                                 #ForBilledTo
        fdist[word_billedto.lower()]+=1
    for word_total in total_value:                                                                                                                     #ForTotal
        fdist[word_total.lower()] += 1
    for word_wt in wt:                                                                                                                                #ForWeight
        fdist[word_wt.lower()]+=1
    for word_custref in cst_ref:                                                                                                           #ForCustomerReference
        fdist[word_custref.lower()]+=1
    for word_ret in ret:                                                                                                                           #ForRetainage
        fdist[word_ret.lower()]+=1
    for word_shippedfrom in shipped_from:                                                                                                        #ForShippedFrom
        fdist[word_shippedfrom.lower()] += 1
    for word_shippedto in shipped_to:                                                                                                              #ForShippedTo
        fdist[word_shippedto.lower()] += 1
    for word_shippeddate in shipped_date:                                                                                                        #ForShippedDate
        fdist[word_shippeddate.lower()] += 1
    for word_cstid in cst_id:                                                                                                                     #ForCustomerId
        fdist[word_cstid.lower()] += 1
    for word_pickedupdate in pickedup_date:                                                                                                       #ForPickUpdate
        fdist[word_pickedupdate.lower()] += 1
    if(fdist["invoiceno"]>=1 or fdist["invoicenumber"]>=1 or fdist["invoice#"]>=1):
        print("INVOICE NUMBER FOUND",inv)
        invoice_number.append(1)
    else:
        print("INVOICE NUMBER NOT FOUND",inv)
        invoice_number.append(0)
    if(fdist["total"]>=1):
        print("TOTAL FOUND",total_value)
        total.append(1)
    else:
        print("TOTAL NOT FOUND",total_value)
        total.append(0)
    if(fdist["description"]>=1):
        print("DESCRIPTION FOUND",des)
        description.append(1)
    else:
        print("DESCRIPTION NOT FOUND",des)
        description.append(0)
    if(fdist["weight"]>=1):
        print("WEIGHT FOUND",wt)
        weight.append(1)
    else:
        print("WEIGHT NOT FOUND",wt)
        weight.append(0)
    if(fdist["qty"]>=1 or fdist["nty"]>=1 or fdist["arty"]>=1):
        print("QUANTITY FOUND",qty)
        quantity.append(1)
    else:
        print("QUANTITY NOT FOUND",qty)
        quantity.append(0)
    if(fdist["rate"]>=1 ):
        print("RATE FOUND",rate)
        unit_price.append(1)
    else:
        print("RATE NOT FOUND",rate)
        unit_price.append(0)
    if(fdist["duedate"]>=1 ):
        print("DUE DATE FOUND",due)
        due_date.append(1)
    else:
        print("DUE DATE NOT FOUND",due)
        due_date.append(0)
    if(fdist["bito"]>=1 or fdist["billto"]>=1 ):
        print("BILL TO FOUND",billed_to)
        bill_to.append(1)
    else:
        print("BILL TO NOT FOUND",billed_to)
        bill_to.append(0)
    if(fdist["refno"]>=1 or fdist["referencenumber"]>=1 or fdist["ref#"]>=1 ):
        print("CUSTOMER REFERENCE FOUND",cst_ref)
        cust_ref.append(1)
    else:
        print("CUSTOMER REFERENCE NOT FOUND",cst_ref)
        cust_ref.append(0)
    if(fdist["retainage"]>=1):
        print("RETAINAGE FOUND",ret)
        retainage.append(1)
    else:
        print("RETAINAGE NOT FOUND",ret)
        retainage.append(0)
    if(fdist["from"]>=1):
        print("SHIPFROM FOUND",shipped_from)
        ship_from.append(1)
    else:
        print("SHIPFROM NOT FOUND",shipped_from)
        ship_from.append(0)
    if(fdist["to"]>1):
        print("SHIPTO FOUND",shipped_to)
        ship_to.append(1)
    else:
        print("SHIPTO NOT FOUND",shipped_to)
        ship_to.append(0)
    if(fdist["terms"]>=1):
        print("SHIPDATE FOUND",shipped_date)
        ship_date.append(1)
    else:
        print("SHIPDATE NOT FOUND",shipped_date)
        ship_date.append(0)
    if(fdist["custref"]>=1):
        print("CUSTOMER ID FOUND",cst_id)
        customer_id.append(1)
    else:
        print("CUSTOMER ID NOT FOUND",cst_id)
        customer_id.append(0)
    if(fdist["date"]>=1 or fdist["pickupdate"]>=1 or (fdist["towingdate"]>=1)):
        print("DATE ID FOUND",pickedup_date)
        pickup_date.append(1)
    else:
        print("DATE NOT FOUND",pickedup_date)
        pickup_date.append(0)
    data = {'TYPE':type,
            'INVOICE NUMBER':invoice_number,
            'DUE DATE':due_date,
            'DESCRIPTION/DETAILS':description,
            'QUANTITY':quantity,
            'UNIT PRICE':unit_price,
            'BILL TO':bill_to,
            'TOTAL':total,
            'WEIGHT':weight,
            'CUSTOMER REFERENCE':cust_ref,
            'RETAINAGE':retainage,
            'FROM':ship_from,
            'TO':ship_to,
            'SHIP DATE':ship_date,
            'CUSTOMER ID':customer_id,
            'PICKUP DATE':pickup_date}
    df = pd.DataFrame(data)
    with open(directory+testCsv, 'a') as f:
        df.to_csv(f, header=False,index=False, sep=',', encoding='utf-8')

#---------------------------------------------------------------------------------------------------------------------------------------------------------
# train = pd.read_csv(directory+trainCsv, header=0)
# test = pd.read_csv(directory+testCsv, header=0)
# print(train.shape)
# print(test.shape)
# print(train.head())
# print(train.describe())
# features = train.iloc[:,1:15]
# labels = train['LABEL']
# x1,x2,y1,y2 =train_test_split(features, labels, random_state=42, train_size =0.3)
# print(x1.shape)
# print(x2.shape)
# print(y1.shape)
# print(y2.shape)
# gnb = GaussianNB()
# KNN = KNeighborsClassifier(n_neighbors=1)
# MNB = MultinomialNB()
# BNB = BernoulliNB()
# LR = LogisticRegression()
# SDG = SGDClassifier()
# SVC = SVC()
# LSVC = LinearSVC()
# NSVC = NuSVC(kernel='rbf',nu=0.01)
# gnb.fit(x1, y1)
# y2_GNB_model = gnb.predict(x2)
# print("GaussianNB Accuracy :", accuracy_score(y2, y2_GNB_model))
#
# KNN.fit(x1,y1)
# y2_KNN_model = KNN.predict(x2)
# print("KNN Accuracy :", accuracy_score(y2, y2_KNN_model))
#
# MNB.fit(x1,y1)
# y2_MNB_model = MNB.predict(x2)
# print("MNB Accuracy :", accuracy_score(y2, y2_MNB_model))
#
# BNB.fit(x1,y1)
# y2_BNB_model = BNB.predict(x2)
# print("BNB Accuracy :", accuracy_score(y2, y2_BNB_model))
#
# LR.fit(x1,y1)
# y2_LR_model = LR.predict(x2)
# print("LR Accuracy :", accuracy_score(y2, y2_LR_model))
#
# SDG.fit(x1,y1)
# y2_SDG_model = SDG.predict(x2)
# print("SDG Accuracy :", accuracy_score(y2, y2_SDG_model))
#
# SVC.fit(x1,y1)
# y2_SVC_model = SVC.predict(x2)
# print("SVC Accuracy :", accuracy_score(y2, y2_SVC_model))
#
# LSVC.fit(x1,y1)
# y2_LSVC_model = LSVC.predict(x2)
# print("LSVC Accuracy :", accuracy_score(y2, y2_LSVC_model))
#
# NSVC.fit(x1,y1)
# y2_NSVC_model = NSVC.predict(x2)
# print("NSVC Accuracy :", accuracy_score(y2, y2_NSVC_model))

#---------------------------------------------------------------------------Classifier--------------------------------------------------------------------
