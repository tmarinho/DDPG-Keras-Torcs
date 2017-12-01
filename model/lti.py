import numpy as np
import scipy.integrate as integrate
from math import sin, cos
class LTI:
    def __init__(self, state_0):
        """ pos = [x,y,z] attitude = [rool,pitch,yaw]
            """
        self.state = state_0
        self.s_dot = 0
        self.hist = []
        self.time = 0.0
        control_frequency = 200 # Hz for attitude control loop
        self.dt = 1.0 / control_frequency


    def reset(self):
        self.state = 0
        self.hist = []
        return self.state

    def state_dot(self, state, t, u, time):
        x1 = state
        b = 1
        a = 5
        self.s_dot = 0.0
        #d =0.014*sin(time*5)
        self.s_dot  = -a*x1 + b*u

        return self.s_dot

    def reward(self,desired_state,u):
        return -10*(self.state - desired_state)**2 - 0.5*u**2

    def update(self, u):
        #saturate u
        u = np.clip(u,-10,10)
        out = integrate.odeint(self.state_dot, self.state, [0,self.dt], args = (u,self.time))
        self.time += self.dt
        #print out
        self.state = out[1]
        self.hist.append(np.array(self.state[0]))

    def step(self, action):
        desired_state = -1.2
        done = False
        #action = agent.do_action(env.state)
        self.update(action)
        reward = self.reward(desired_state, action)
        if abs(self.state) > 3 or abs(self.s_dot)>50:
            done = True

        return self.state[0], reward, done, {}
