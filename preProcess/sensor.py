import pandas as pd
sen=pd.read_csv("..//oriData//on-street-parking-bay-sensors.csv",index_col=False,keep_default_na=False)
sen=sen.drop(["parking_zone", "last_updated"],axis=1)
sen=sen[["bay_id","st_marker_id","status","location","lat","lon"]]
baylist=[i for i in range(len(sen))]
# sen["bay_id"].replace("",1000,inplace=True)
sen["bay_id"]=pd.DataFrame(baylist)
sen.to_csv("..//data//On-street_Parking_Bay_Sensors.csv",index=False)
