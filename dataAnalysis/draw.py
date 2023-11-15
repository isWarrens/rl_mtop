from matplotlib import pyplot as plt
import pandas as pd
fileName="..//logs//2023-08-15_23-13-25_352625//scores.csv"
figName="..//fig//"
df=pd.read_csv(fileName,index_col=None,header=None)
stepList=df.loc[:,1]
rewardList=df.loc[:,3]
disRewardList=df.loc[:,2]
epochList=[i for i in range(len(df))]

plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(epochList,rewardList)
plt.title("Reward")
plt.xlabel("epoch")
plt.ylabel("reward")

plt.subplot(2,1,2)
plt.plot(epochList,disRewardList)
plt.title("discounted Reward")
plt.xlabel("epoch")
plt.ylabel("discounted reward")
plt.tight_layout()
plt.savefig(figName+"2023-08-15_23-13-25_352625.jpg")
plt.show()
