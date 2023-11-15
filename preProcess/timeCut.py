'''
只保留2.08号一天的数据
'''
import pandas as pd
import datetime
# sensorData=pd.read_csv('..//oriData//On-street_Car_Parking_Sensor_Data_-_2017.csv')
# for i in range(len(sensorData)):
#     sensorData["DeviceId"].iloc[i]=int(sensorData["DeviceId"].iloc[i].replace(",",""))
# sensorData['ArrivalTime1']=pd.to_datetime(sensorData['ArrivalTime'])
#
# sensorData['day']=sensorData['ArrivalTime1'].dt.date
#
#
# sensorData=sensorData[sensorData['day']==datetime.date(2017,2,8)]
# sensorData=sensorData.drop(['day'],axis=1)
# sensorData=sensorData.drop(['ArrivalTime1'],axis=1)
# sensorData.to_csv('..\\data\\On-street_Car_Parking_Sensor_Data_-_2017_02.csv',index=False)
# print("保留一天")

'''
只保留2.08号早八点到达的数据
'''
import pandas as pd
import datetime
sensorData=pd.read_csv('..\\data\\On-street_Car_Parking_Sensor_Data_-_2017_02.csv')
# for i in range(len(sensorData)):
#     sensorData["DeviceId"].iloc[i]=int(sensorData["DeviceId"].iloc[i].replace(",",""))

sensorData['ArrivalTime1']=pd.to_datetime(sensorData['ArrivalTime'])
sensorData['hour']=sensorData['ArrivalTime1'].dt.hour
sensorData['minute']=sensorData['ArrivalTime1'].dt.minute


sensorData=sensorData[sensorData['hour']<19]
sensorData=sensorData[sensorData['hour']>7]
sensorData=sensorData[sensorData['minute']%10>=9]
sensorData=sensorData.drop(['hour'],axis=1)
sensorData=sensorData.drop(['minute'],axis=1)
sensorData=sensorData.drop(['ArrivalTime1'],axis=1)
sensorData.to_csv('..\\data\\On-street_Car_Parking_Sensor_Data_-_2017_02_08.csv',index=False)
print("保留一小时")