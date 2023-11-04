##1.environment.py
        while True:
            if self.next_event:
                times = [
                  nsmallest(1, self.violation_times)[0][0] if len(self.violation_times) > 0 else np.inf,
                   nsmallest(1, self.departures)[0][0] if len(self.departures) > 0 else np.inf,
                   nsmallest(1, self.delayed_arrival)[0][0] if len(self.delayed_arrival) > 0 else np.inf,
                    self.next_event['ArrivalTime'],
                ]
           else:
               times = [
                   nsmallest(1, self.violation_times)[0][0] if len(self.violation_times) > 0 else np.inf,
                    nsmallest(1, self.departures)[0][0] if len(self.departures) > 0 else np.inf,
                    nsmallest(1, self.delayed_arrival)[0][0] if len(self.delayed_arrival) > 0 else np.inf,

                ]
            if min(times) > self.time:
                break
            arg = np.argmin(times)
增加self.next_event的判断，因为有报错显示self.next_event不存在的情况


##2.create.py
        data[idx] = datetime.strptime(data[idx], "%m/%d/%Y %I:%M:%S %p").strftime("%s")
strftime("%s")无法识别"%s"格式，environment中同样存在这种问题
##3.on-street-parking-bay-sensors.csv
    baylist=[i for i in range(len(sen))]
    sen["bay_id"]=pd.DataFrame(baylist)
由于官网下载的数据集不带bay_id，导致这里缺失信息，经过判断认为

    1.通过其他文件匹配bay_id难以实现，因为没有对应的数据
    2.bay_id在数据库处理和后续算法中没有发挥作用
因此简单赋值
##4.Road_Corridor.csv
        for the_geom, rd_seg_id, marker_id, bay_id, meter_id, rd_seg_dsc, last_edit in tqdm(reader):
            # point = shapely_loads(the_geom).centroid
            point=the_geom.split(',')
            lat, lon = float(point[0]), float(point[1])
            cursor.execute("INSERT OR IGNORE INTO locations (marker, lat, lon) VALUES(\"%s\", \"%f\", \"%f\")" % (marker_id, lat, lon))
源代码是计算得到对应的geo_point
我们下载的数据有geo_point和geo_shape两列，因此drop了geo_shape，直接使用了geo_point
##5.on-street-car-park-bay-restrictions.csv
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
 该文件主要有两个问题
 
    1.根据create.py,原数据应当是"%H:%M:%S"格式，我们下载的数据为“0001-01-02T04:34:43+08:06”形式，因此切片处理
    2.我们下载的数据大量空白，我这里直接把空白地方补上了两个默认时间
##6.DQN&gamma
doubleDQN&gamma==1
>Traceback (most recent call last):
  File "F:/RL_FAIR/Semi-Markov-Reinforcement-Learning-for-Stochastic-Resource-Collection-master/Semi-Markov-Reinforcement-Learning-for-Stochastic-Resource-Collection-master/main.py", line 723, in <module>
    train_top({
  File "F:/RL_FAIR/Semi-Markov-Reinforcement-Learning-for-Stochastic-Resource-Collection-master/Semi-Markov-Reinforcement-Learning-for-Stochastic-Resource-Collection-master/main.py", line 677, in train_top
    experiment(mdp, params, prob=None)
  File "F:/RL_FAIR/Semi-Markov-Reinforcement-Learning-for-Stochastic-Resource-Collection-master/Semi-Markov-Reinforcement-Learning-for-Stochastic-Resource-Collection-master/main.py", line 540, in experiment
    core.learn(n_steps=params['evaluation_frequency'],
  File "C:\ProgramData\Anaconda3\envs\rltop\lib\site-packages\mushroom_rl\core\core.py", line 68, in learn
    self._run(n_steps, n_episodes, fit_condition, render, quiet)
  File "C:\ProgramData\Anaconda3\envs\rltop\lib\site-packages\mushroom_rl\core\core.py", line 118, in _run
    return self._run_impl(move_condition, fit_condition, steps_progress_bar,
  File "C:\ProgramData\Anaconda3\envs\rltop\lib\site-packages\mushroom_rl\core\core.py", line 146, in _run_impl
    self.agent.fit(dataset)
  File "C:\ProgramData\Anaconda3\envs\rltop\lib\site-packages\mushroom_rl\algorithms\value\dqn\dqn.py", line 87, in fit
    self._fit(dataset)
  File "C:\ProgramData\Anaconda3\envs\rltop\lib\site-packages\mushroom_rl\algorithms\value\dqn\dqn.py", line 118, in _fit_prioritized
    q = reward + self.mdp_info.gamma * q_next
**ValueError: operands could not be broadcast together with shapes (128,2) (128,)**
