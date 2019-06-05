import pytesseract
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from PIL import Image,ImageEnhance
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
#-------------------------------------------------------------------EndOfimports-----------------------------------------------------------------------------
Extension=".jpg"
FILE="\\"+"test3"
EnhancedExtension="_enhanced"
#-------------------------------------------------------------------EndOfFileInputs--------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd=r'C:\\Users\\Vishal.Chauhan\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'                                #setting path
FileToBeEnhanced=Image.open(r'C:\\Users\\Vishal.Chauhan\\Desktop\\tessert'+FILE+Extension)                                                    #IMAGE LOCATION
# enhancer = ImageEnhance.Sharpness(FileToBeEnhanced)
enhancer = ImageEnhance.Color(FileToBeEnhanced)
enhanced_im = enhancer.enhance(2.75)
enhanced_im.save(r'C:\\Users\\Vishal.Chauhan\\Desktop\\tessert'+FILE+EnhancedExtension+Extension)
t=Image.open(r'C:\\Users\\Vishal.Chauhan\\Desktop\\tessert'+FILE+EnhancedExtension+Extension)                                              #OpenEnhancedImage
text=pytesseract.image_to_string(t,lang="eng")                                                                                              #ExtractionOfText
print(text)
text1=re.sub(r'[^a-zA-Z0-9#/_\s]+', '',text)                                                                                     #Removing special characters
token=word_tokenize(text1)                                                                                                                      #tokenisation
#print(token)                                                                                                                               #tokenised output
stop_words = set(stopwords.words('english'))
filtered_sentence = [w for w in token if not w in stop_words]                                                                      #stopwords   - a ,the , is
filtered_sentence = []
for w in token:
    if w not in stop_words:
        filtered_sentence.append(w)
print(filtered_sentence)                                                                                                                    #stopwords output
words = [word for word in filtered_sentence if word.isalpha()]                                                                                #For Words Only
numbers = [word for word in filtered_sentence if word.isnumeric()]                                                                          #For Numbers Only
print(words)
print(numbers)
amount=numbers[-1]                                                                                                                         #To Extract Amount
valueFetched=int(amount)/100
print(valueFetched)
#-----------------------------------------------------------EnterValueToDataFrame----------------------------------------------------------------------------
type=[0];invoice_number=[];due_date=[];description=[];quantity=[];unit_price=[];bill_to=[];total=[];weight=[];cust_ref=[];retainage=[];ship_from=[];ship_to=[];ship_date=[];customer_id=[];pickup_date=[]
my_list_string=''.join(words)
fdist=FreqDist()
inv=re.findall(r"\BINVOICE#|\BINVOICENo|\BINVOICENO|\BINVOICENUMBER",my_list_string)
due=re.findall(r"\BDUEDATE|\BD..D..e|\BDATED",my_list_string)
des=re.findall(r"\BDes......on",my_list_string)
qty=re.findall(r"\BQty|\BQ.....TY|\B.ty|\Barty",my_list_string)
rate=re.findall(r"\BRATE",my_list_string)
billed_to=re.findall(r"\BBILLTO|\BBillto|\BBiTo|\BB...to",my_list_string)
total_value=re.findall(r"\BTotal|\BTOTAL",my_list_string)
wt=re.findall(r"\BW....t",my_list_string)
cst_ref=re.findall(r"\BRefNo|\BREFNO|\BR..N.|\BREFERENCENUMBER|\BREF#",my_list_string)
ret=re.findall(r"\BRETAINAGE|\BRET....GE",my_list_string)
shipped_from=re.findall(r"\BFROM",my_list_string)
shipped_to=re.findall(r"\BTO",my_list_string)
shipped_date=re.findall(r"\BTERMS",my_list_string)
cst_id=re.findall(r"\BCustRef",my_list_string)
pickedup_date=re.findall(r"\BDate|\BDATE|\BPICKUPDATE",my_list_string)
for word_inv in inv:                                                            #ForInvoiceNumber
    fdist[word_inv.lower()]+=1
for word_due in due:                                                            #ForDueDate
    fdist[word_due.lower()]+=1
for word_des in des:                                                            #ForDescription
    fdist[word_des.lower()]+=1
for word_qty in qty:                                                            #ForQuantity
    fdist[word_qty.lower()]+=1
for word_rate in rate:                                                          #ForRate
    fdist[word_rate.lower()]+=1
for word_billedto in billed_to:
    fdist[word_billedto.lower()]+=1
for word_total in total_value:                                                  #ForTotal
    fdist[word_total.lower()] += 1
for word_wt in wt:                                                              #ForWeight
    fdist[word_wt.lower()]+=1
for word_custref in cst_ref:
    fdist[word_custref.lower()]+=1
for word_ret in ret:
    fdist[word_ret.lower()]+=1
for word_shippedfrom in shipped_from:
    fdist[word_shippedfrom.lower()] += 1
for word_shippedto in shipped_to:
    fdist[word_shippedto.lower()] += 1
for word_shippeddate in shipped_date:
    fdist[word_shippeddate.lower()] += 1
for word_cstid in cst_id:
    fdist[word_cstid.lower()] += 1
for word_pickedupdate in pickedup_date:
    fdist[word_pickedupdate.lower()] += 1
if(fdist["invoiceno"]>=1 or fdist["invoicenumber"]>=1):
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
if(fdist["date"]>=1 or fdist["pickupdate"]>=1):
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
with open(r'C:\Users\Vishal.Chauhan\Desktop\tessert\dataset.csv', 'a') as f:
    df.to_csv(f, header=False,index=False, sep=',', encoding='utf-8')

#---------------------------------------------------------------------------------------------------------------------------------------------------------
# my_dataframe=pd.read_csv()
# my_dataframe.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
#---------------------------------------------------------------------------Classifier--------------------------------------------------------------------


#------------------------------------------------------------------------ImportingDataSet-----------------------------------------------------------------

