import pandas as pd
import tqdm
res=pd.read_csv("..//oriData/on-street-car-park-bay-restrictions.csv",keep_default_na=False,index_col=False)
endtime="0001-01-01T18:00:00+08:00"
starttime="0001-01-01T08:00:00+08:00"
for i in range(1,7):
    res['EndTime'+str(i)].replace("",endtime,inplace=True)
    res['StartTime' + str(i)].replace("", starttime, inplace=True)
# print(res['EndTime6'])
for i in tqdm.tqdm(range(len(res))):
    for j in range(1,7):
        # print(res['EndTime' + str(j)].iloc[i])
        res['EndTime'+str(j)].iloc[i]=res['EndTime'+str(j)].iloc[i][11:19]
        res['StartTime'+str(j)].iloc[i] = res['StartTime'+str(j)].iloc[i][11:19]
# print(res['EndTime1'])
res.to_csv("..//data/On-street_Car_Park_Bay_Restrictions.csv",index=False)