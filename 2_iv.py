#this code fits an ellipse and a circle to the triangulation 
#data, outputs the loss for each and saves a plot in the
#plot.png file in the same directory
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import gmean
from scipy.optimize import minimize
import matplotlib.pyplot as plt

df=pd.read_csv("./../data/01_data_mars_triangulation.csv")
#print(df)

data=df.iloc[:,4:8]

e_deg=df.iloc[:,4:5].values
e_min=df.iloc[:,5:6].values
m_deg=df.iloc[:,6:7].values
m_min=df.iloc[:,7:8].values

earth_helio=(np.pi)/180 * (e_deg + (e_min/60))
mars_geo= (np.pi)/180 * (m_deg + (m_min/60))

opt_r=1.5773209091612428

x_list=[]
y_list=[]
def xy_val(val1):
  rad=val1
  for r in range(int(len(earth_helio.tolist())/2)):
    c=r*2
    theta1=earth_helio[c]
    theta2=earth_helio[c+1]
    alpha1=mars_geo[c]
    alpha2=mars_geo[c+1]
    x=(np.cos(theta1)*np.tan(alpha1)-np.sin(theta1)-np.cos(theta2)*np.tan(alpha2)+np.sin(theta2))/(np.tan(alpha1)-np.tan(alpha2))
    y=np.sin(theta1) + np.tan(alpha1)*(x - np.cos(theta1))
    x_list.append(float(x))
    y_list.append(float(y))
  #print(x_list,y_list)

xy_val(opt_r)

def fit_circle_tri():
    l=0
    for i in range(len(x_list)):
      Ri = np.sqrt((x_list[i])**2 + (y_list[i])**2)
      l += (opt_r - Ri)**2
    return(l)
l=fit_circle_tri()
print("total optimized loss for circle : ",l)

def fit_ellipse_tri(args):
  x1 = args[0]
  y1 = args[1]
  l = args[2]
  loss=0
  for i in range(len(x_list)):
    d1=np.sqrt((x_list[i])**2 + (y_list[i])**2 )
    d2=np.sqrt((x1-x_list[i])**2 + (y1-y_list[i])**2)
    loss += (d1 + d2 - l)**2
  return loss

from scipy.optimize import basinhopping
xy=[4, -4, 1]
minimizer_kwargs = {"method": "BFGS"}
opt_elip1 = basinhopping(fit_ellipse_tri, xy, minimizer_kwargs=minimizer_kwargs)
print("total optimized loss for ellipse: ",opt_elip1["fun"])
#print("optimized x and y values : ", opt_elip1["x"])

# Example focii and sum-distance
a1 = 0
b1 = 0
a2 = opt_elip1["x"][0]
b2 = opt_elip1["x"][1]
c = opt_elip1["x"][2]

# Compute ellipse parameters
a = c / 2                                # Semimajor axis
x0 = (a1 + a2) / 2                       # Center x-value
y0 = (b1 + b2) / 2                       # Center y-value
f = np.sqrt((a1 - x0)**2 + (b1 - y0)**2) # Distance from center to focus
b = np.sqrt(a**2 - f**2)                 # Semiminor axis
phi = np.arctan2((b2 - b1), (a2 - a1))   # Angle betw major axis and x-axis

# Parametric plot in t
resolution = 1000
t = np.linspace(0, 2*np.pi, resolution)
x = x0 + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
y = y0 + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)

# Plot ellipse
plt.plot(x, y)
plt.scatter(x_list,y_list)
#plt.scatter(m_x,m_y)
#plt.Circle((0, 0), opt_r,fill=False)
circle1=plt.Circle((0,0),opt_r,color='r',fill=False)
plt.gcf().gca().add_artist(circle1)
plt.plot(a1, b1, 'bo')
plt.plot(a2, b2, 'bo')

plt.axis('equal')
plt.savefig('plot.png')
