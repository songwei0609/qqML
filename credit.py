import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import re as re

import seaborn as sns

train_df = pd.read_csv('cs-training.csv')
test_df = pd.read_csv('cs-test.csv')

print (train_df.info())
print (train_df.describe())

train_df.rename(columns={'Unnamed: 0':'ID'}, inplace=True)
test_df.rename(columns={'Unnamed: 0':'ID'}, inplace=True)

P = train_df.groupby('SeriousDlqin2yrs')['ID'].count().reset_index()
P['Percentage'] = 100 * P['ID'] / P['ID'].sum()
print (P)


#plt.figure()
#sns.countplot('SeriousDlqin2yrs',data=train_df)
#plt.show()

#corr = train_df.corr()
#plt.figure(figsize=(19, 15))
#sns.heatmap(corr, annot=True, fmt='.2g')

# 去掉98和96两个点，再查看相关性如何
def replace98and96(column):
    new = []
    newval = column.median()
    for i in column:
        if (i == 96 or i == 98):
            new.append(newval)
        else:
            new.append(i)
    return new

train_df['NumberOfTime30-59DaysPastDueNotWorse'] = replace98and96(train_df['NumberOfTime30-59DaysPastDueNotWorse'])
train_df['NumberOfTimes90DaysLate'] = replace98and96(train_df['NumberOfTimes90DaysLate'])
train_df['NumberOfTime60-89DaysPastDueNotWorse'] = replace98and96(train_df['NumberOfTime60-89DaysPastDueNotWorse'])

test_df['NumberOfTime30-59DaysPastDueNotWorse'] = replace98and96(test_df['NumberOfTime30-59DaysPastDueNotWorse'])
test_df['NumberOfTimes90DaysLate'] = replace98and96(test_df['NumberOfTimes90DaysLate'])
test_df['NumberOfTime60-89DaysPastDueNotWorse'] = replace98and96(test_df['NumberOfTime60-89DaysPastDueNotWorse'])

#'''
plt.figure(figsize=(19, 12))
train_df[['NumberOfTime30-59DaysPastDueNotWorse',
          'NumberOfTime60-89DaysPastDueNotWorse',
          'NumberOfTimes90DaysLate']].boxplot()
plt.show()

corr = train_df.corr()
plt.figure(figsize=(19, 15))
sns.heatmap(corr, annot=True, fmt='.2g')
plt.show()
#'''

test_df.loc[test_df['age'] == 0, 'age'] = test_df['age'].median()

# 按照退休年龄划分数据集
working = train_df.loc[(train_df['age'] >= 18) & (train_df['age'] <= 60)]
senior = train_df.loc[(train_df['age'] > 60)]
working_income_mean = working['MonthlyIncome'].mean()
senior_income_mean = senior['MonthlyIncome'].mean()

# 退休与否差距不大，对收入的空数据填充
train_df['MonthlyIncome'] = train_df['MonthlyIncome'].replace(np.nan,train_df['MonthlyIncome'].mean())

# 对空值用中位数填充
train_df['NumberOfDependents'].fillna(train_df['NumberOfDependents'].median(), inplace=True)


test_df.loc[test_df['age'] == 0, 'age'] = test_df['age'].median()
test_df['MonthlyIncome'] = test_df['MonthlyIncome'].replace(np.nan,test_df['MonthlyIncome'].mean())
test_df['NumberOfDependents'].fillna(test_df['NumberOfDependents'].median(), inplace=True)

# 设定训练集和测试集的输入输出
X = train_df.drop(['SeriousDlqin2yrs', 'ID'],axis=1)
y = train_df['SeriousDlqin2yrs']
W = test_df.drop(['SeriousDlqin2yrs', 'ID'],axis=1)
z = test_df['SeriousDlqin2yrs']

# 用线性回归模型
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=111)

# 调用线性回归函数，C为正则化系数，l1表示L1正则化
logit = LogisticRegression(random_state=111, solver='sag', penalty='l2', class_weight='balanced', C=1.0, max_iter=500)

# 标准化拟合
scaler = StandardScaler().fit(X_train)

# 标准化X_train 和X_test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 线性回归拟合
logit.fit(X_train_scaled, y_train)

# 输入训练集，返回每个样本对应到每种分类结果的概率
logit_scores_proba = logit.predict_proba(X_train_scaled)

# 返回分类1的概率
logit_scores = logit_scores_proba[:,1]

# 画图
def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(12,10))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1], "k--") # 画直线做参考
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive r0ate")

# roc_curve根据分类结果和分类概率，返回false positive rage和true positive rate
fpr_logit, tpr_logit, thresh_logit = roc_curve(y_train, logit_scores)

# 画图
plot_roc_curve(fpr_logit,tpr_logit)
print ('AUC Score : ', (roc_auc_score(y_train,logit_scores)))


# 验证测试集，测试分类结果概率分布
logit_scores_proba_val = logit.predict_proba(X_test_scaled)

# 分类结果为1的概率
logit_scores_val = logit_scores_proba_val[:,1]

# roc_curve根据分类结果和分类概率，返回false positive rage和true positive rate
fpr_logit_val, tpr_logit_val, thresh_logit_val = roc_curve(y_test, logit_scores_val)

# 画图
plot_roc_curve(fpr_logit_val,tpr_logit_val)
print ('AUC Score :', (roc_auc_score(y_test,logit_scores_val)))

plt.show()

