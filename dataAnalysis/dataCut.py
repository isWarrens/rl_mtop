import pandas as pd
import numpy as np
fileNme = "..//data//On-street_Car_Parking_Sensor_Data_-_2017_02_08.csv"
resultName="..//fig//On-street_Car_Parking_Sensor_Data_-_2017_02_08.csv"
df = pd.read_csv(fileNme,index_col=0)
df = df[df["Area"].isin(['Docklands','Southbank','Queensberry'])]
df.to_csv(resultName)
