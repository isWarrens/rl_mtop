import pandas as pd
import numpy as np
import statistics
class AreaInfo():
    def __init__(self,areaList=None):
        self.areaList=areaList
        self.areaVioList=None
        self.areaCapList=None
        self.areaCapDict=None
        self.areaVioDict=None
        self.deviceAreaDict=None

        self.areaCapList = [0]*len(self.areaList)
        self.areaVioList = [0]*len(self.areaList)
        self.areaCapStepList=[0]*len(self.areaList)
        self.areaCapDict =dict(zip(self.areaList,[i for i in range(len(self.areaList))]))
        #key:area名称 value:0到n-1，即areaList的index
        self.areaVioDict =dict(zip(self.areaList,[i for i in range(len(self.areaList))]))

        self.deviceAreaDict={}
        self.lastRewardValue=None
        self.rewardValue = None

    def areaCapStd(self):
        capList=[]
        for vioNum,capNum in zip(self.areaVioList,self.areaCapList):
            if vioNum==0:
                vioNum=1
            capRate=capNum/vioNum
            capList.append(capRate)
        areaCapStd=statistics.stdev(capList)
        return areaCapStd

    def recordAreaOfDeviceVio(self,device):
        area=self.deviceAreaDict[device]
        self.areaVioList[self.areaVioDict[area]] += 1

    def recordAreaOfDeviceCap(self,device):
        area = self.deviceAreaDict[device]
        self.areaCapList[self.areaCapDict[area]] += 1
        self.areaCapStepList[self.areaCapDict[area]] += 1

    def caculStepReward(self):
        reward=0
        capNum = 0
        vioNum = 0
        for i in range(len(self.areaCapList)):
            capNum += self.areaCapList[i]
            vioNum += self.areaVioList[i]
        if vioNum > 0:
            reward = capNum / vioNum
        else:
            reward = capNum / 1
        reward-=AreaInfo.areaCapStdTool(self.areaCapList,self.areaVioList)
        reward-=self.lastRewardValue
        self.lastRewardValue=reward+self.lastRewardValue
        return reward



    def device2Area(self,event):

        for i in range(len(event)):

            if event["DeviceId"].iloc[i] not in self.deviceAreaDict:
                self.deviceAreaDict[event["DeviceId"].iloc[i]] = event["Area"].iloc[i]

    def initial(self):
        self.areaCapList = [0] * len(self.areaList)
        self.areaVioList = [0] * len(self.areaList)
        self.lastRewardValue=0

    def initialStep(self):
        self.areaCapStepList = [0] * len(self.areaList)

    @staticmethod
    #计算抓捕率的标准差
    def areaCapStdTool(areaCapList,areaVioList):
        capList = []
        for vioNum, capNum in zip(areaVioList, areaCapList):
            if vioNum == 0:
                vioNum = 1
            capRate = capNum / vioNum
            capList.append(capRate)
        areaCapStd = statistics.stdev(capList)
        return areaCapStd

    # @staticmethod
    # #根据每一步的reward计算reward value
    # def caculRewardStep(areaReward):
    #     areaRewardVal = 0
    #     for i in range(len(areaReward[0])):
    #         areaRewardVal += areaReward[0][i]
    #     areaVioList = areaReward[2]
    #     areaCapList = areaReward[0]
    #     areaRewardVal -= AreaInfo.areaCapStdTool(areaCapList,areaVioList)
    #     return areaRewardVal

    @staticmethod
    def caculRewardTotal(areaReward):
        capNum = 0
        vioNum = 0
        for i in range(len(areaReward[1])):
            capNum += areaReward[1][i]
            vioNum += areaReward[2][i]
        if vioNum > 0:
            reward = capNum / vioNum
        else:
            reward = capNum / 1
        reward -= AreaInfo.areaCapStdTool(areaReward[1], areaReward[2])
        return reward