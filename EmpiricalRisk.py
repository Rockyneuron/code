"""Emprirical Risk lecture exercise. 
    Unit 2-4
"""

import numpy as np

#-------------------------Calculate Empirical risk using hinge loss-------------------------------

feature_vectors=np.array([[1,0,1],[1,1,1],[1,1,-1],[-1,1,1]])
values=np.array([2,2.7,-0.7,2])
theta=np.array([0,1,2])


def hinge_loss_full(feature_matrix, labels, theta, theta_0=0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    z_hinge_loss=labels-(np.dot(feature_matrix,theta.T)+theta_0)
    hinge_loss=  np.average([1-z if z<1 else 0 for z in z_hinge_loss]) 
    return  hinge_loss

def hinge_loss_square_loss(feature_matrix, labels, theta, theta_0=0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    z_hinge_loss=labels-(np.dot(feature_matrix,theta.T)+theta_0)
    hinge_loss=  np.average([z**2/2 for z in z_hinge_loss]) 
    return  hinge_loss


empirical_risk=hinge_loss_full(feature_vectors,values,theta)
print('emprirical risk with hinge loss is equal to {}'.format(empirical_risk))
empirical_risk=hinge_loss_square_loss(feature_vectors,values,theta)
print('emprirical risk with hinge loss with squared loss as loss function is equal to {}'.format(empirical_risk))


#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits import mplot3d
style.use('dark_background')
#-------------------------Mapping feature vectors in new dimensions-------------------------------
"""Code for unit 2-6 problem
"""

def random_interval_generator(upper_limit,lower_limit,length):
    vector=np.zeros(length)
    for t in range(length):
     vector[t]=np.random.uniform() * (upper_limit - lower_limit) + lower_limit  
    return vector

samples=1000
#Sample feature vectors
points_1_x=random_interval_generator(0.5,-0.5,samples)
points_1_y=random_interval_generator(0.2,-0.7,samples)
points_1_labels=np.ones(samples)*2

points_2_x=np.append(random_interval_generator(-1,1,int(samples/2)),random_interval_generator(0.6,1,int(samples/2)))
points_2_y=np.append(random_interval_generator(0.25,1,int(samples/2)),random_interval_generator(-1,1,int(samples/2)))
points_2_labels=np.ones(samples)*10


fig1 = plt.figure()
ax1 = plt.axes()
ax1.scatter(points_1_x,points_1_y,points_1_labels)
ax1.scatter(points_2_x,points_2_y,points_2_labels)

x_1=points_1_x
y_1=points_1_y
x_2=points_2_x
y_2=points_2_y



x_2_1=[x_1,y_1,x_1*y_1]
x_2_2=[x_2,y_2,x_2*y_2]
x_3_1=[x_1,y_1,x_1**2+y_1**2]
x_3_2=[x_2,y_2,x_2**2+y_2**2]

x_4_1=[x_1,y_1,x_1**2+2*y_1**2]
x_4_2=[x_2,y_2,x_2**2+2*y_2**2]


fig = plt.figure()
ax = plt.axes(projection='3d')

# ax.scatter(x_2_1[0],x_2_1[1],x_2_1[2],marker=3)
# ax.scatter(x_2_2[0],x_2_2[1],x_2_2[2],marker=3)

# ax.scatter(x_3_1[0],x_3_1[1],x_3_1[2],marker=3)
# ax.scatter(x_3_2[0],x_3_2[1],x_3_2[2],marker=3)

ax.scatter(x_4_1[0],x_4_1[1],x_4_1[2],marker=3)
ax.scatter(x_4_2[0],x_4_2[1],x_4_2[2],marker=3)
# ax.plot3D(x_4[0],x_4[1],x_4[2],'.')
plt.show()



# %%
