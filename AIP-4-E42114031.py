import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 导包获取糖尿病数据集
from sklearn.datasets import load_diabetes  
data_diabetes = load_diabetes()    
print(data_diabetes)  

data =  data_diabetes['data']
target = data_diabetes['target']
feature_names = data_diabetes['feature_names']
df =  pd.DataFrame(data,columns = feature_names)

from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y =  train_test_split(data,target,train_size =0.8)


# 建立模型
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# 训练数据
model.fit(train_X,train_Y)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

# 评估模型
model.score(train_X,train_Y)

pred_Y = model.predict(test_X)
print(mean_squared_error(test_Y, pred_Y))
pred_Y = model.predict(train_X)
print(mean_squared_error(train_Y, pred_Y))

df.columns
plt.figure(figsize=(40,30),dpi=50)
for i,col in enumerate(df.columns):  
    train_X = df.loc[:,col].values.reshape(-1,1)    

    train_Y = target
    linear_model = LinearRegression()    # 构建模型
    linear_model.fit(train_X,train_Y)    #训练模型
    score = linear_model.score(train_X,train_Y)   # 评估模型

    axes = plt.subplot(2,5,i+1)
    plt.scatter(train_X,train_Y)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.3)

    k =  linear_model.coef_     # 回归系数
    b =  linear_model.intercept_   # 截距
    x = np.linspace(train_X.min(),train_X.max(),100)
    y = k * x + b
# 作图
    plt.plot(x,y,c='red')
    axes.set_title(col + ':' + str(score))
plt.show()

data_diabetes = load_diabetes(return_X_y = 1)
L = list(zip(data_diabetes[0], data_diabetes[1]))
import random
random.shuffle(L)
L1 = L[0:(int)(len(L) * 0.8)]
L2 = L[(int)(len(L) * 0.8):]
tags=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
linear_model.fit([record[0] for record in L1], [record[1] for record in L1])

for i in range(10):
	plt.bar(i, linear_model.coef_[i])

plt.title("各特征权重柱状图")       #柱状图标题
plt.xlabel("特征类型")         #X轴名称
plt.ylabel("权重")         #Y轴名称
plt.xticks([i for i in range(10)],tags,rotation=90)
plt.show()                    # 显示柱状图