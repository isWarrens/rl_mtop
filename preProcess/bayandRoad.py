import pandas as pd

bayLoc=pd.read_csv("..//oriData//on-street-parking-bays.csv")
road=pd.read_csv("..//oriData//Road_Corridor.csv")
# bayLoc=bayLoc.drop(['Geo Shape'],axis=1)

# bayLoc.to_csv('data/bay_locations.csv',index=False)
road=road.drop(['Geo Shape'],axis=1)
#以上是删除geo shape的信息
road=road[["gisid","seg_id","Geo Point","street_id","status_id","seg_part","str_type","dtupdate","poly_area","seg_descr"]]
road.columns=["GISID","SegID","the_geom","StreetID","StatusID","SegPart","StrType","DTUpdate","PolyArea","SegDescr"]
#以上调整road文件的column
road.to_csv('..//data/Road_Corridor.csv',index=False)