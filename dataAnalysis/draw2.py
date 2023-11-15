'''
utility,fairness,reward,discounted reward
'''
from matplotlib import pyplot as plt
import pandas as pd
import statistics
import numpy as np
fileName = "..//logs//2023-09-14_16-14-26_352625//scores.csv"
#读取csv文件地址
figNameHead="..//fig//"
figName = "01"
#设置图片名称
df = pd.read_csv(fileName,index_col=None,header=None)
stepList = df.loc[:,1]
rewardList = df.loc[:,3]
disRewardList = df.loc[:,2]
resListList = df.loc[:,6]
utilityList = df.loc[:,4]
fairList = df.loc[:,5]



epochList=[i for i in range(len(df))]

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.plot(stepList,rewardList)
plt.title("Reward")
plt.xlabel("step")
plt.ylabel("reward")

plt.subplot(2,2,2)
plt.plot(stepList,disRewardList)
plt.title("discounted Reward")
plt.xlabel("step")
plt.ylabel("discounted reward")

plt.subplot(2,2,3)
plt.plot(stepList,utilityList)
plt.title("utility")
plt.xlabel("step")
plt.ylabel("utility")

plt.subplot(2,2,4)
plt.plot(stepList,fairList)
plt.title("fairness")
plt.xlabel("step")
plt.ylabel("fairness")

plt.tight_layout()
plt.savefig(figNameHead+figName+".jpg")
plt.show()

