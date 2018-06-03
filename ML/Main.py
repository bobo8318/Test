import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import numpy as np

"""
data = bytes(urllib.parse.urlencode({'word': 'hello'}), encoding='utf8')
response = urllib.request.urlopen('http://httpbin.org/post', data=data)
"""

url = "http://www.baidu.com"
headers = {
    #伪装一个火狐浏览器
    "User-Agent":'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)',
    "host":'httpbin.org'
}
dict = {
    "name":"Germey"
}
data = bytes(urllib.parse.urlencode(dict),encoding="utf8")
req = urllib.request.Request(url=url)
response = urllib.request.urlopen(req)

"""print(response.read().decode("utf-8"))

soup = BeautifulSoup('<div>hello</div>','html.parser')
if(soup.div is not None):
    print(soup.div.string)
"""
"""
soup = BeautifulSoup(response, 'lxml')
array = soup.select('a')
for i in range(0, len(array)):

    attrs = array[i].attrs
    print(attrs["href"],array[i].text)


    pass
"""
""" 正则表达式
import re

line = "Cats are smarter2than dogs and cats"

matchObj = re.match( r'(.*) are (.*?) (.*)', line, re.M|re.I)

if matchObj:
    print ("matchObj.group() : ", matchObj.group())
    print ("matchObj.group(1) : ", matchObj.group(1))
    print ("matchObj.group(2) : ", matchObj.group(2))
    print ("matchObj.group(2) : ", matchObj.group(3))
else:
    print ("No match!!")

pattern = re.compile('[a-z\d]')
result = pattern.findall('a s 3SiOPdj#@23awe')
settest = set()
settest.add('girl')
print(settest)
"""

"""

list = [[1,2,3],[4,5,6]]
nplist = np.array(list)
np.random.randint(1,10,3)

import matplotlib.pyplot as plt

x = np.linspace(-np.pi,np.pi,256,endpoint=True)
c,s = np.cos(x),np.sin(x)
plt.figure(1)
plt.plot(x,c,color="blue",linewidth=1.0,linestyle="-",label="cos",alpha=0.5)
plt.plot(x,s,"r*",label="sin")
plt.title("cos & sin")
plt.show()
"""
def autoNorm(dataset):
    """ newvalue = oldvalue-minvalue/maxvalue-minvalue"""
    """一列的最大值 最小值 范围 向量"""
    minvalues = dataset.min(0)
    maxvalues = dataset.max(0)
    ranges = maxvalues - minvalues
    print( minvalues)
    """同大小的0矩阵"""
    normDataSet = np.zeros(np.shape(dataset))
    """行数"""
    m = normDataSet.shape[0]

    normDataSet = dataset - np.tile(minvalues,(m,1))
    print( np.tile(minvalues,(m,1)))
    print(normDataSet)
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return  normDataSet,ranges,minvalues
pass
"""
oldset = np.random.randint(0,2,[3,4])
print(oldset)
featlist = [example[2] for example in oldset ]
print("featlist:")
print(set(featlist))

import NaiveBayes
lostOpOsts ,listClasses = NaiveBayes.loadDataSet()

myVocabList = NaiveBayes.createVocabList(lostOpOsts)

trainMat = []
for postingdoc in lostOpOsts:
    trainMat.append(NaiveBayes.setOfWords2Vec(myVocabList,postingdoc))


a,b,c = NaiveBayes.trainNB0(trainMat,listClasses)

import re
regEx = re.compile(r'\W*')
liststr = regEx.split("hello,how are you")
print(liststr)

import Logistic
dataset = np.mat(np.random.randint(0,2,[3,4]))
weight = np.ones((4,1))

for j in range(5):
    dataindex=range(2)
    for i in range(2):
        print(len(dataindex))
    pass
pass
"""
"""
dataset = np.mat(np.random.randint(0,2,[3,4]))
print(dataset)

alphas = np.mat(np.zeros([3, 1]))
labelMat = np.mat([1,0,1]).transpose()
a = np.multiply(alphas,labelMat).T
b = (dataset*dataset[1,:].T)
print(a)
print(b)
print(a*b)
"""
x = -5
y = -5
#print(x is y)

#from MySql import MySqlTest

#MySqlTest.dataBaseTest()

'''
import pandas as pd
from pylab import *
xnums = np.arange(4)
ynums = np.arange(5)
x,y = np.meshgrid(xnums,ynums)

left = pd.DataFrame({"key":["x","y"],"value":["1","2"]})
right = pd.DataFrame({"key":["x","z"],"value":["3","4"]})
pd.merge(left,right,on="key",how="left")# 相当于sql 表连接 以left为准 取key值相等的行 将 left right连接 right没有的值为nan
#how= inner out 内连接 left 左连接 right 右连接
df3 = pd.DataFrame({"A":["a","b","c","b"],"B":list(range(4))})
df3.groupby("A").sum()# sql 中的group by

# reshape

df4 = pd.DataFrame()
#pd.pivot_table(df4,values="D",index=["A","B"],columns="C")

# time series
dates = pd.date_range("20170301",periods=10,freq="S")#时间序列
# graph
ts = pd.Series(np.random.randn(1000),index=pd.date_range("20170301",periods=1000))
ts = ts.cumsum()
ts.plot()
show()
'''
#func = lambda x:np.exp(-x),0,np.inf
def Before(request,kargs):
    print('before')

def After(request,kargs):
    print('after')


def Filter(before_func,after_func):
    def outer(main_func):# main_func 调用函数 Index
        def wrapper(request,kargs):# main_func带参数调用 也就是实际调运行ndex（）

            before_result = before_func(request,kargs)#前置方法
            if(before_result != None):
                return before_result;

            main_result = main_func(request,kargs)#实际方法
            if(main_result != None):
                return main_result;

            after_result = after_func(request,kargs)#后置方法
            if(after_result != None):
                return after_result;

        return wrapper#outer返回
    return outer#Filter 返回

@Filter(Before, After)
def Index(request,kargs):
    print('index')

Index(1,2)