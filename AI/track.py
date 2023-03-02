import numpy as np

class KF(object):
    def __init__(self, dt, u_x, u_y, sigma_u_dot, sigma_x, sigma_y):
        """
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param sigma_u_dot: process noise magnitude
        :param sigma_x: standard deviation of the measurement in x-direction
        :param sigma_y: standard deviation of the measurement in y-direction
        """
        
        # Define sampling time
        self.dt = dt

        # Define the  control input variables
        self.u = np.matrix([[u_x],[u_y]])

        # Intial State
        self.x = np.matrix([[0], [0], [0], [0]])

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        #Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * sigma_u_dot ** 2

        #Initial Measurement Noise Covariance
        self.R = np.matrix([[sigma_x ** 2, 0],
                            [0, sigma_y ** 2]])

        #Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):

        # Update states
        # x_k =Ax_(k-1) + Bu_(k-1)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Extrapolate error covariance
        # P= A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]


    def update(self, z):


        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))

        # Update error covariance matrix
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P   #Eq.(13)
        return self.x[0:2]


class noiseFilt(object):

    def __init__(self, input_value,window_size = 10):
        """
        :param dt: sampling time (time for 1 cycle)
        """
        self.input_value=input_value
        self.window_size=window_size
        self.window = np.empty(1)

    def median(self):
        self.window = np.insert(self.window, 0, self.input_value)
        self.window = self.window[:self.window_size]
        sorted_window = sorted(self.window)

        l=len(sorted_window)
        # print(len(sorted_window))

        if (l % 2) == 0:
            # Even
            filtered_output = (sorted_window[int(l / 2)] + sorted_window[int((l / 2) - 1)])/2
        else:
            # Odd window size
            filtered_output = sorted_window[int((l - 1) / 2)]

        return filtered_output