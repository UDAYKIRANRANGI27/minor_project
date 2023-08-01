
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse

#import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

# Create your views here.
from Remote_User.models import ClientRegister_Model,Credit_Card_Fraud_Detection,detection_accuracy,detection_results,detection_ratio


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def Find_CreditCard_ResultsDetails(request):

    obj = detection_results.objects.all()
    return render(request, 'SProvider/Find_CreditCard_ResultsDetails.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = detection_results.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_DetectedCredit_CardFraudDetails(request):
    obj =Credit_Card_Fraud_Detection.objects.all()
    return render(request, 'SProvider/View_DetectedCredit_CardFraudDetails.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="TrainedData.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Credit_Card_Fraud_Detection.objects.all()

    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1
        ws.write(row_num, 0, 'Fraud_Cases', font_style)
        ws.write(row_num, 1, my_row.Fraud_Cases, font_style)
        ws.write(row_num, 2, 'Valid_Transactions', font_style)
        ws.write(row_num, 3, my_row.Valid_Transactions, font_style)

    wb.save(response)
    return response

def Train_Test_DataSets(request):
    detection_accuracy.objects.all().delete()
    detection_results.objects.all().delete()
    detection_ratio.objects.all().delete()
    Credit_Card_Fraud_Detection.objects.all().delete()

    # load dataset
    data = pd.read_csv("creditcard.csv")
    data.head(10)
    # describing the data
    print(data.shape)
    print(data.describe())
    # imbalance in the data
    fraud = data[data['Class'] == 1]
    valid = data[data['Class'] == 0]
    outlierFraction = len(fraud) / float(len(valid))
    print(outlierFraction)
    print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
    print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))
    # the amount details for fraudulent transaction
    fraud.Amount.describe()
    # the amount details for normal transaction
    valid.Amount.describe()
    # plotting the correlation matrix
    corrmat = data.corr()
    # separating the X and the Y values
    X = data.drop(['Class'], axis=1)
    Y = data["Class"]
    print(X.shape)
    print(Y.shape)
    # getting just the values for the sake of processing
    # (its a numpy array with no columns)
    xData = X.values
    yData = Y.values
    # training and testing data bifurcation
    from sklearn.model_selection import train_test_split
    # split the data into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)
    # building the Random Forest Classifier
    print("RandomForestClassifier")
    # random forest model creation
    rfc = RandomForestClassifier()
    rfc.fit(xTrain, yTrain)
    # predictions
    yPred = rfc.predict(xTest)
    n_outliers = len(fraud)
    n_errors = (yPred != yTest).sum()
    print("The model used is Random Forest classifier")
    acc = accuracy_score(yTest, yPred)
    print("The accuracy is {}".format(acc))
    prec = precision_score(yTest, yPred)
    print("The precision is {}".format(prec))
    rec = recall_score(yTest, yPred)
    print("The recall is {}".format(rec))
    f1 = f1_score(yTest, yPred)
    print("The F1-Score is {}".format(f1))
    print('Classification report:\n', classification_report(yTest, yPred))
    print('Confusion matrix:\n', confusion_matrix(y_true=yTest, y_pred=yPred))

    detection_accuracy.objects.create(names="Random Forest Classifier", ratio=acc * 100)
    detection_results.objects.create(model_name="Random Forest Classifier", precision1=prec,recall=rec,F1_Score=f1)

    print("Decision Tree Classifier")
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(xTrain, yTrain)
    y_predicted = model.predict(xTest)

    acc1 = accuracy_score(yTest, y_predicted)
    print("The accuracy is {}".format(acc1))

    prec1 = precision_score(yTest, y_predicted)
    print("The precision is {}".format(prec))
    rec1 = recall_score(yTest, y_predicted)
    print("The recall is {}".format(rec))
    f11 = f1_score(yTest, y_predicted)
    print("The F1-Score is {}".format(f1))


    print('Classification report:\n', classification_report(yTest, y_predicted))
    print('Confusion matrix:\n', confusion_matrix(y_true=yTest, y_pred=y_predicted))

    Credit_Card_Fraud_Detection.objects.create(Fraud_Cases=len(data[data['Class'] == 1]), Valid_Transactions=len(data[data['Class'] == 0]))
    detection_accuracy.objects.create(names="Decision Tree Classifier", ratio=acc1 * 100)
    detection_results.objects.create(model_name="Decision Tree Classifier", precision1=prec1, recall=rec1, F1_Score=f11)


    ratio = ""
    kword = 'Fraud_Cases'
    print(kword)
    obj = Credit_Card_Fraud_Detection.objects.get()
    fraud_cases=int(obj.Fraud_Cases)
    ratio = (fraud_cases / 100005) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Valid_Transactions'
    print(kword1)
    obj1 = Credit_Card_Fraud_Detection.objects.get()
    Valid_Transactions = int(obj1.Valid_Transactions)
    ratio1 = (Valid_Transactions / 100005) * 100
    if ratio1 != 0:
        detection_ratio.objects.create(names=kword1, ratio=ratio1)

    obj = detection_accuracy.objects.all()
    return render(request,'SProvider/Train_Test_DataSets.html', {'objs': obj})