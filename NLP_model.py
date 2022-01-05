"""
创建时间：2022/1/4 17:47
开发者：啊振
努力尽今夕
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import warnings
import os

warnings.filterwarnings('ignore')

# 批量读取和合并文本数据集

def read_text(path,text_list):
    """
    :param path: 必选参数，文件夹路径
    :param text_list: 必选参数，文件夹 path 下的所有 .txt 文件名
    :return:返回值
            features 文本特征数据，以列表形式返回
            labels 分类标签，以列表形式返回
    """
    features, labels = [], []
    for text in text_list:
        if text.split('.')[-1] == 'txt':
            try:
                with open(path + text, encoding='gbk') as fp:
                    features.append(fp.read()) # 特征
                    labels.append(path.split('\\')[-2]) # 标签
            except Exception as erro:
                print('\n>>>发现错误，正在输出错误信息。。。\n', erro)
    return features, labels


def merge_text(train_or_text, label_name):
    """

    :param train_or_text: 必选参数，train 训练数据集 or text 测试数据集
    :param label_name: 必选参数，分类标签的名字
    :return: 返回值
            merge_features 合并好的所有特征数据，以列表形式返回
            merge_labels 合并好的所有分类标签数据，以列表形式返回
    """
    print('\n>>>文本读取和合并程序已经启动，请稍等。。。。')
    merge_features, merge_labels = [], []
    for name in label_name:
        path = 'D:\\NLP自然语言处理\\text_classification\\text classification\\' + train_or_text + '\\' + name + '\\'
        print(path)
        text_list = os.listdir(path)
        print(text_list)
        features, labels = read_text(path=path, text_list=text_list)
        merge_features += features # 特征
        merge_labels += labels # 标签
    return merge_features, merge_labels


# 获取训练集
train_or_text = 'train'
label_name = ['女性','体育','校园','文学']
X_train, y_train = merge_text(train_or_text, label_name)

train_or_text = 'test'
label_name = ['女性','体育','校园','文学']
X_test, y_test = merge_text(train_or_text, label_name)
print(X_test)
