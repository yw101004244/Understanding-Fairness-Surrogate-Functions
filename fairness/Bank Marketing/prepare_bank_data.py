import os,sys
sys.path.append(r'c:\Users\yao\Desktop\fair-classification-master\fair_classification') # the code for fair classification is in this directory
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from collections import Counter

def load_bank_data():

    df = pd.read_csv(r'c:\Users\yao\Desktop\fair-classification-master\disparate_impact\Bank Marketing\bank\bank-full.csv',header=0,sep=';')



    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)




    # preprocess
    attrs = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'] # all attributes
    int_attrs = ['age', 'balance', 'pdays', 'duration','day','month','previous','campaign'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['age'] # the fairness constraints will be used for this feature


    X = []
    y = []
    x_control = {}


    for i in range(df.shape[0]):
        if df.iloc[i,0] > 60 or df.iloc[i,0] < 25:
            df.iloc[i,0] = 0
        else:
            df.iloc[i,0] = 1

    x_control['age'] = df.iloc[:,0]



    class_label = df.loc[:,'y']

    class_label.replace('no',-1,inplace=True)
    class_label.replace('yes',1,inplace=True)
    y = class_label



    # label encoder and one hot encoder
    LE = LabelEncoder()
    OH = OneHotEncoder()

    for attr in attrs: # all attributes
        if attr not in int_attrs: # not in integer attributes means categorical attributes, so deal with this column 
            df[attr] = LE.fit_transform(df[attr]) # using LabelEncoder
            
            df_np = df[attr].values
            OH.fit_transform(df_np.reshape(-1,1))
            after_onehot = OH.transform(df_np.reshape(-1,1)).toarray()
            
            after_onehot = pd.DataFrame(after_onehot)
            df = pd.concat([df,after_onehot],axis=1)

            df.drop(columns=attr)

    df['month'] = LE.fit_transform(df['month'])


    # StandardScaler
    standard_scaler_attrs = ['balance', 'pdays', 'duration','day','month','previous','campaign']
    scaler = StandardScaler()
    df[standard_scaler_attrs] = scaler.fit_transform(df[standard_scaler_attrs])


    y = y.values

    # To compute the term, some preparation is leaved in the main program

    return df, y, x_control











def load_bank_additional_data():

    df = pd.read_csv(r'C:\Users\yao\Desktop\understanding\fairness\Bank Marketing\bank-additional\bank-additional-full.csv',header=0,sep=';')
    
    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)


    # df = df[:1000]


    attrs = ['age','job','marital','education','default','housing','loan','contact','month','day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'] # all attributes
    int_attrs = ['age', 'pdays', 'duration','month','day_of_week','previous','campaign','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['age'] # the fairness constraints will be used for this feature


    X = []
    y = []
    x_control = {}


    for i in range(df.shape[0]):
        if df.iloc[i,0] > 60 or df.iloc[i,0] < 25:
            df.iloc[i,0] = 0
        else:
            df.iloc[i,0] = 1

    x_control['age'] = df.iloc[:,0]



    class_label = df.loc[:,'y']

    class_label.replace('no',-1,inplace=True)
    class_label.replace('yes',1,inplace=True)
    y = class_label





    LE = LabelEncoder()
    OH = OneHotEncoder()


    for attr in attrs: # all attributes
        if attr not in int_attrs: # categorical attributes, so deal with this column 
            df[attr] = LE.fit_transform(df[attr]) # using LabelEncoder
            
            df_np = df[attr].values
            OH.fit_transform(df_np.reshape(-1,1))
            after_onehot = OH.transform(df_np.reshape(-1,1)).toarray()
            
            after_onehot = pd.DataFrame(after_onehot)
            df = pd.concat([df,after_onehot],axis=1)

            df.drop(columns=attr)

    df['month'] = LE.fit_transform(df['month'])
    df['day_of_week'] = LE.fit_transform(df['day_of_week'])





    scaler = StandardScaler()
    standard_scaler_attrs = ['pdays', 'duration','month','day_of_week','previous','campaign',
                 'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m', 'nr.employed'] 
    
    df[standard_scaler_attrs] = scaler.fit_transform(df[standard_scaler_attrs])


    y = y.values


    return df, y, x_control







def load_bank_balanced_data():
    df = pd.read_csv('https://raw.githubusercontent.com/rafiag/DTI2020/main/data/bank.csv')

    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)




    attrs = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'] # all attributes
    int_attrs = ['age', 'balance', 'pdays', 'duration','day','month','previous','campaign'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['age'] # the fairness constraints will be used for this feature


    X = []
    y = []
    x_control = {}


    for i in range(df.shape[0]):
        if df.iloc[i,0] > 60 or df.iloc[i,0] < 25:
            df.iloc[i,0] = 0
        else:
            df.iloc[i,0] = 1

    x_control['age'] = df.iloc[:,0]



    class_label = df.loc[:,'deposit']


    class_label.replace('no',-1,inplace=True)
    class_label.replace('yes',1,inplace=True)
    y = class_label





    LE = LabelEncoder()
    OH = OneHotEncoder()


    for attr in attrs: # all attributes
        if attr not in int_attrs: # not in integer attributes means categorical attributes, so deal with this column 
            df[attr] = LE.fit_transform(df[attr]) # using LabelEncoder
            
            df_np = df[attr].values
            OH.fit_transform(df_np.reshape(-1,1))
            after_onehot = OH.transform(df_np.reshape(-1,1)).toarray()
            
            after_onehot = pd.DataFrame(after_onehot)
            df = pd.concat([df,after_onehot],axis=1)

            df.drop(columns=attr)

    df['month'] = LE.fit_transform(df['month'])


    scaler = StandardScaler()
    df[int_attrs] = scaler.fit_transform(df[int_attrs])


    # df = df.drop('duration',axis=1)
    df = df.drop('age',axis=1)


    y = y.values
    X = df


    X = X.drop('deposit',axis=1)

    X = X.values


    return X,y,x_control









def load_kaggle_bank_data():

    df = pd.read_csv(r'c:\Users\yao\Desktop\fair-classification-master\disparate_impact\Bank Marketing\kaggle_bank\bank.csv')

    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)


    attrs = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'] # all attributes
    int_attrs = ['age', 'balance', 'pdays', 'duration','day','month','previous','campaign'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['age'] # the fairness constraints will be used for this feature


    X = []
    y = []
    x_control = {}


    for i in range(df.shape[0]):
        if df.iloc[i,0] > 60 or df.iloc[i,0] < 25:
            df.iloc[i,0] = 0
        else:
            df.iloc[i,0] = 1

    x_control['age'] = df.iloc[:,0]



    class_label = df.loc[:,'deposit']

    class_label.replace('no',-1,inplace=True)
    class_label.replace('yes',1,inplace=True)
    y = class_label





    LE = LabelEncoder()
    OH = OneHotEncoder()


    for attr in attrs: # all attributes
        if attr not in int_attrs: # categorical attributes, so deal with this column 
            df[attr] = LE.fit_transform(df[attr]) # using LabelEncoder
            
            df_np = df[attr].values
            OH.fit_transform(df_np.reshape(-1,1))
            after_onehot = OH.transform(df_np.reshape(-1,1)).toarray()
            
            after_onehot = pd.DataFrame(after_onehot)
            df = pd.concat([df,after_onehot],axis=1)

            df.drop(columns=attr)

    df['month'] = LE.fit_transform(df['month'])


    scaler = StandardScaler()
    df[int_attrs] = scaler.fit_transform(df[int_attrs])


    # df = df.drop('duration',axis=1)
    df = df.drop('age',axis=1)


    y = y.values
    X = df

    X = X.drop('deposit',axis=1)

    X = X.values


    return X,y,x_control







def test():
    from sklearn.model_selection import train_test_split,cross_val_score
    from sklearn.linear_model import LogisticRegression

    X, y, x_control = load_bank_additional_data()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=1)

    logreg = LogisticRegression()
    model_logreg = logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# test()