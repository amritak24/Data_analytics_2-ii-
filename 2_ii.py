#this code outputs the radius after optimizing the
#loss for the data points in triangulation data set
#it also prints the total loss after optimization.
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import gmean
from scipy.optimize import minimize

df=pd.read_csv("./../data/01_data_mars_triangulation.csv")
#print(df)

data=df.iloc[:,4:8]

e_deg=df.iloc[:,4:5].values
e_min=df.iloc[:,5:6].values
m_deg=df.iloc[:,6:7].values
m_min=df.iloc[:,7:8].values

earth_helio=(np.pi)/180 * (e_deg + (e_min/60))
mars_geo= (np.pi)/180 * (m_deg + (m_min/60))

def rad(val1):
  rad=val1
  #b=val2
  radlist=[]
  for r in range(int(len(earth_helio.tolist())/2)):
    #print(int(len(earth_helio.tolist())/2))
    #print(r)
    c=r*2
    theta1=earth_helio[c]
    theta2=earth_helio[c+1]
    alpha1=mars_geo[c]
    alpha2=mars_geo[c+1]
    x=(np.cos(theta1)*np.tan(alpha1)-np.sin(theta1)-np.cos(theta2)*np.tan(alpha2)+np.sin(theta2))/(np.tan(alpha1)-np.tan(alpha2))
    y=np.sin(theta1) + np.tan(alpha1)*(x - np.cos(theta1))
    radius=np.sqrt(x*x + y*y)
    radlist.append(radius)
  return radlist

def minimize_mean(xy):
  r=xy
  radius_list=rad(r)
  s=0
  for i in radius_list:
    s+=((i-r)*(i-r))
  return s

xy=[2.5]
params = minimize(minimize_mean, xy, bounds=[(0, None)])
print("total optimized loss : ",params["fun"])
print("optimized value for radius : ", params["x"][0])
