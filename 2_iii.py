#this code optimizes the values for the coefficients 
#of the plane and outputs the inclination in degrees
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

opt_r=1.5773209091612428

earth_helio=(np.pi)/180 * (e_deg + (e_min/60))
mars_geo= (np.pi)/180 * (m_deg + (m_min/60))

df2=pd.read_csv("./../data/01_data_mars_opposition.csv")
#print(df2)

m_lat_deg=df2["LatDegree"].values
m_lat_min=df2["LatMinute"].values
lat_rad=(np.pi)/180 * (m_lat_deg + (m_lat_min/60))
#print(lat_rad)

data_wrt_sun=df2[['ZodiacIndex','Degree','Minute','Second']]
#print(data_wrt_sun)
sun_data_list=data_wrt_sun.values.tolist()
#print(sun_data_list)
mars_long_deg=[]
mars_long_rad=[]
for i in sun_data_list:
  p=float(i[0]*30+i[1]+i[2]/60+i[3]/3600)
  mars_long_deg.append(p)
  mars_long_rad.append(p*math.pi/180)

m_x=[]
m_y=[]
m_z=[]
phi=[]
#opt_r=1.57
#list_lat_rad=(list(lat_rad))
for i in range(len(lat_rad)):
  p=np.arctan((opt_r-1)/opt_r)*np.tan(lat_rad[i])
  phi.append(p)
  m_x.append(np.cos(p)*np.cos(mars_long_rad[i]))
  m_y.append(np.cos(p)*np.sin(mars_long_rad[i]))
  m_z.append(np.sin(p))
#print(phi)
#print(m_x)
#print(m_y)
#print(m_z)

def minimize_plane(xy):
  a= xy[0]
  b= xy[1]
  c= xy[2]
  d= xy[3]
  loss=0
  for i in range(len(m_x)):
    loss += np.abs(a*m_x[i] + b*m_y[i] + c*m_z[i] + d)/(np.sqrt(a*a + b*b + c*c))
  return loss
  

from scipy.optimize import basinhopping
xy=[4, -4, 200, -0.2]
minimizer_kwargs = {"method": "BFGS"}
opt_abc = basinhopping(minimize_plane, xy, minimizer_kwargs=minimizer_kwargs)
#opt_abc = minimize(minimize_plane, xy)
#print(opt_abc)
#print(opt_abc)
#print("total optimized loss : ",opt_abc["fun"])
#print("optimized values for a,b,c,d", opt_abc["x"])

opt_a=opt_abc["x"][0]
opt_b=opt_abc["x"][1]
opt_c=opt_abc["x"][2]
print("inclination is ",np.arccos(opt_c/(np.sqrt(opt_a*opt_a + opt_b*opt_b + opt_c*opt_c))) * (180/np.pi),"degrees")
