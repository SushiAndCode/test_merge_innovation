'''
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime


def rollEstimation(df):
    #measurements = np.asarray([(399,293),(403,299),(409,308),(416,315),(418,318),(420,323),(429,326),(423,328),(429,334),(431,337),(433,342),(434,352),(434,349),(433,350),(431,350),(430,349),(428,347),(427,345),(425,341),(429,338),(431,328),(410,313),(406,306),(402,299),(397,291),(391,294),(376,270),(372,272),(351,248),(336,244),(327,236),(307,220)])
    measurements = np.asarray(df['Gy'])
    dt = (datetime.strptime(df['time'][1], "%H:%M:%S.%f") - datetime.strptime(df['time'][0], "%H:%M:%S.%f")).total_seconds()
    initial_state_mean = [0, measurements[0]]
 
    transition_matrix = [[1, dt],
                        [0, 1]]

    observation_matrix = [0, 1]
    
    transition_cov = [[1, 0],
                        [0, 1]]

    observation_cov = 1

    transition_offset = [0.5*dt**2, dt]
    
    kf = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean,
                    transition_covariance = transition_cov, 
                    observation_covariance = observation_cov)

    kf = kf.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

    plt.figure(1)
    times = range(measurements.shape[0])
    plt.plot(times, measurements, 'bo',
            times, smoothed_state_means, 'r--',)
    plt.show()



import numpy as np
from numpy.linalg import inv

x_observations = np.array([4000, 4260, 4550, 4860, 5110])
v_observations = np.array([280, 282, 285, 286, 290])

z = np.c_[x_observations, v_observations]

# Initial Conditions
a = 2  # Acceleration
v = 280
t = 1  # Difference in time

# Process / Estimation Errors
error_est_x = 20
error_est_v = 5

# Observation Errors
error_obs_x = 25  # Uncertainty in the measurement
error_obs_v = 6

def prediction2d(x, v, t, a):
    A = np.array([[1, t],
                  [0, 1]])
    X = np.array([[x],
                  [v]])
    B = np.array([[0.5 * t ** 2],
                  [t]])
    X_prime = A.dot(X) + B.dot(a)
    return X_prime


def covariance2d(sigma1, sigma2):
    cov1_2 = sigma1 * sigma2
    cov2_1 = sigma2 * sigma1
    cov_matrix = np.array([[sigma1 ** 2, cov1_2],
                           [cov2_1, sigma2 ** 2]])
    return np.diag(np.diag(cov_matrix))


# Initial Estimation Covariance Matrix
P = covariance2d(error_est_x, error_est_v)
A = np.array([[1, t],
              [0, 1]])

# Initial State Matrix
X = np.array([[z[0][0]],
              [v]])
n = len(z[0])

for data in z[1:]:
    X = prediction2d(X[0][0], X[1][0], t, a)
    # To simplify the problem, professor
    # set off-diagonal terms to 0.
    P = np.diag(np.diag(A.dot(P).dot(A.T)))

    # Calculating the Kalman Gain
    H = np.identity(n)
    R = covariance2d(error_obs_x, error_obs_v)
    S = H.dot(P).dot(H.T) + R
    K = P.dot(H).dot(inv(S))

    # Reshape the new data into the measurement space.
    Y = H.dot(data).reshape(n, -1)

    # Update the State Matrix
    # Combination of the predicted state, measured values, covariance matrix and Kalman Gain
    X = X + K.dot(Y - H.dot(X))

    # Update Process Covariance Matrix
    P = (np.identity(len(K)) - K.dot(H)).dot(P)

print("Kalman Filter State Matrix:\n", X)

'''   
import numpy as np
from numpy.linalg import inv
from datetime import datetime
import matplotlib.pyplot as plt


def rollEstimation(df):

    # Initial Conditions
    x0 = 0
    v0 = v_ang = df['Gy'][0]
    # Data
    # arm of the IMU sensor placement wrt ground
    arm = 0.65 # [m]
    a_ang = df['Ay'] / arm  # Angular acceleration (lateral acc (from IMU) / radius) [rad/s^2]
    v_ang = df['Gy']        # Angular velocity (from IMU) [rad/s]       
    dt = (datetime.strptime(df['time'][1], "%H:%M:%S.%f") - datetime.strptime(df['time'][0], "%H:%M:%S.%f")).total_seconds()


    # Process / Estimation Errors
    error_est_x = 1
    error_est_v = 1

    # Observation Errors
    error_obs_x = 1 # Uncertainty in the measurement
    error_obs_v = 1

    def prediction2d(x, v, t, a):
        A = np.array([[1, dt],
                    [0, 1]])
        X = np.array([x, v])
        B = np.array([[0.5 * dt ** 2],
                    [dt]])
        X_prime = A.dot(X.reshape((2,1))) + B.dot(a)
        return X_prime


    def covariance2d(sigma1, sigma2):
        cov1_2 = sigma1 * sigma2
        cov2_1 = sigma2 * sigma1
        cov_matrix = np.array([[sigma1 ** 2, cov1_2],
                            [cov2_1, sigma2 ** 2]])
        return np.diag(np.diag(cov_matrix))


    # Initial Estimation Covariance Matrix
    P = covariance2d(error_est_x, error_est_v)
    A = np.array([[1, dt],
                [0, 1]])

    # Initial State Matrix
    X = np.array([x0,
                v0])
    n = 2
    X.reshape(n, -1)
    df['theta'] = 0
    df['thetaDot'] = 0

    for i in range(len(df['Gy'])):

        X = prediction2d(float(X[0]), float(df.loc[i,'Gy']), dt, a_ang[i])
        H = np.array([0, 1])
        mix = A.dot(P).dot(H.reshape(n, -1))
        P = np.diag(np.diag(A.dot(P).dot(A.T)))
        
        # Calculating the Kalman Gain
        
        R = error_obs_v**2
        S = H.dot(P).dot(H.T) + R
        K = mix.dot(1/S)

        # Reshape the new data into the measurement space.
        Y = H.dot([0, df.loc[i,'Gy']]) #.reshape(n, -1)

        # Update the State Matrix
        # Combination of the predicted state, measured values, covariance matrix and Kalman Gain
        X = X + K.dot(Y - H.dot(X)).reshape(n, -1)

        df.loc[i, 'theta'] = X[0]
        df.loc[i, 'thetaDot'] = X[1]

        
        # Update Process Covariance Matrix
        P = (np.identity(len(K)) - K.dot(H.reshape(1,2))).dot(P)
