from random import uniform
import numpy as np
from math import radians, sin, cos


class Creature():

    def __init__(self, settings, parameters=None, stats=None, name=None, color=None):
        """
        settings : dictionary
            simulation settings
        parameters : dictionary
            "weights" and "biases" of NN parameters
        stats : dictionary
            creature stats (mass, etc...)
        name : string
            creature identifier
        """
        # Variables
        self.settings = settings
        self.parameters = parameters
        self.stats = stats
        self.name = name
        self.fitness = 0
        self.color = (0, 255, 0) if color is None else color

        # Position
        self.x = uniform(settings['x_min'], settings['x_max'])  # position (x)
        self.y = uniform(settings['y_min'], settings['y_max'])  # position (y)
        self.r = uniform(0, 360)  # orientation   [0, 360]
        self.dr = uniform(0, 360) # rotational speed (degrees per second)
        self.v = uniform(0, settings['v_max'])  # velocity      [0, v_max]
        #self.v = 0
        #self.dr = 0

        # NN
        self.brain = Model(settings['nodes'], parameters)
        self.settings = settings

        # NN output
        self.nn_force_forward = 0 # fraction of forward force applied
        self.nn_force_radial = 0 # fraction of radial force applied

        # NN input
        self.d_food = 100  # distance to nearest food
        self.r_food = 0  # orientation to nearest food
        self.d_org = 100  # distance to nearest  organism
        self.r_org = 0  # orientation to nearest organism
        self.d_pred = 100  # distance to nearest predator
        self.r_pred = 0  # orientation to nearest organism

        # Stats
        if stats is None:
            self.stats = {}
            self.stats['strength'] = 100 # [N = kg m / s^2] Max force that can be applied
            self.stats['mass'] = 10 # kg
            self.stats['density'] = 100 # kg / m^2
            area = self.stats['mass'] / self.stats['density']
            self.stats['radius'] = np.sqrt(area / np.pi)
            self.stats['torque'] = 0 # [kg m^2 / s^2] = [ n * m ]
            self.stats['moment_of_inertia'] = self.stats['mass'] * self.stats['radius'] ** 2 # [kg m^2]
            self.stats['pred'] = False

    def think(self):
        df = 4 * self.d_food / self.settings['x_max']
        do = 4 * self.d_org / self.settings['x_max']
        dp = 4 * self.d_pred / self.settings['x_max']
        if self.stats['pred']:
            self.nn_force_forward, self.nn_force_radial = \
                self.brain.predict(np.array([self.r_org, self.r_pred, self.r_food, do, dp, df]).reshape(6, 1)).T[0]
                # self.brain.predict(np.array([self.r_food, self.r_pred, df, dp]).reshape(4, 1)).T[0]
        else:
            self.nn_force_forward, self.nn_force_radial = \
                self.brain.predict(np.array([self.r_food, self.r_org, self.r_pred, df, do, dp]).reshape(6, 1)).T[0]
                # self.brain.predict(np.array([self.r_food, self.r_pred, df, dp]).reshape(4, 1)).T[0]

    # UPDATE HEADING
    def update_r(self, settings):
        F = self.nn_force_radial * self.stats['strength']
        torque = F * self.stats['radius']
        alpha = torque / self.stats['moment_of_inertia'] # angular acceleration
        self.dr += alpha * settings['dt']
        if self.stats['pred']:
            self.dr = min(self.dr, settings['dr_max']*0.8)
        else:
            self.dr = min(self.dr, settings['dr_max'])
        self.r += self.dr * settings['dt']
        self.r = self.r % 360
        self.r += self.nn_force_radial * settings['dr_max'] * settings['dt']
        self.r = self.r % 360

    # UPDATE VELOCITY
    def update_vel(self, settings):
        F = self.nn_force_forward * self.stats['strength']
        a = F / self.stats['mass']
        self.v += a * settings['dt']
        if self.v < 0: self.v = 0
        if self.stats['pred']:
            if self.v > settings['v_max']*1.2: self.v = settings['v_max']
        else:
            if self.v > settings['v_max']: self.v = settings['v_max']

    # UPDATE POSITION
    def update_pos(self, settings):
        dx = self.v * cos(radians(self.r)) * settings['dt']
        dy = self.v * sin(radians(self.r)) * settings['dt']
        self.x += dx
        self.y += dy

    def respawn(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.fitness -= 5
        if self.fitness < 0: self.fitness = 0

class organism():

    def __init__(self, settings, parameters=None, name = None, color=None):
        # Position
        self.x = uniform(settings['x_min'], settings['x_max'])  # position (x)
        self.y = uniform(settings['y_min'], settings['y_max'])  # position (y)
        self.r = uniform(0, 360)  # orientation   [0, 360]
        self.v = uniform(0, settings['v_max'])  # velocity      [0, v_max]
        self.dv = uniform(-settings['dv_max'], settings['dv_max'])  # dv
        self.nn_dr = 0
        self.nn_dv = 0

        # Stats
        self.name = name
        self.fitness = 0
        self.energy = 1000
        self.color = (0, 255, 0) if color is None else color

        # Environment
        self.d_food = 100  # distance to nearest food
        self.r_food = 0  # orientation to nearest food
        self.d_org = 100 # distance to nearest other organism
        self.r_org = 0 # orientation to nearest other organism

        self.brain = Model(settings['nodes'], parameters)
        self.settings = settings
        #self.brain = build_brain(1,5) if brain is None else brain

    def think(self):
        df = 4 * self.d_food / self.settings['x_max']
        do = 4 * self.d_org / self.settings['x_max']
        #self.nn_dv, self.nn_dr = self.brain.predict(np.array([self.r_food, self.d_food, self.r_org, self.d_org]).reshape(1,4))[0]
        self.nn_F, self.nn_dr = \
            self.brain.predict(np.array([self.r_food, self.r_org, df, do]).reshape(4, 1)).T[0]

    # UPDATE HEADING
    def update_r(self, settings):
        self.r += self.nn_dr * settings['dr_max'] * settings['dt']
        self.r = self.r % 360

    # UPDATE VELOCITY
    def update_vel(self, settings):
        self.v += self.nn_dv * settings['dv_max'] * settings['dt']
        if self.v < 0: self.v = 0
        if self.v > settings['v_max']: self.v = settings['v_max']

    # UPDATE POSITION
    def update_pos(self, settings):
        dx = self.v * cos(radians(self.r)) * settings['dt']
        dy = self.v * sin(radians(self.r)) * settings['dt']
        self.x += dx
        self.y += dy

    def respawn(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.fitness -= 5


class Model():

    def __init__(self, nodes, parameters=None):
        # nodes : list = number of nodes in each layer
        #     starting with the input and ending with the output.
        self.nodes = nodes
        self.n_layers = len(self.nodes) - 1

        if parameters is None: #build random parameters
            self.parameters = {'weights': [], 'biases': []}
            for i in range(self.n_layers):
                self.parameters['weights'].append(
                    np.random.uniform(-1, 1, (self.nodes[i+1], self.nodes[i])))
                self.parameters['biases'].append(
                    np.random.uniform(-1, -1, (self.nodes[i+1], 1))
                )
        else:
            self.parameters = parameters

    def predict(self, X):
        A = X
        af = lambda x: np.tanh(x) # activation function
        relu = lambda x: np.maximum(x, 0, x)
        for i in range(self.n_layers - 1):
            A = relu(np.dot(self.parameters['weights'][i], A) + self.parameters['biases'][i])
        A = af(np.dot(self.parameters['weights'][-1], A) + self.parameters['biases'][-1])
        return A

def get_pred_stats():
    stats = {}
    stats['strength'] = 300  # [N = kg m / s^2] Max force that can be applied
    stats['mass'] = 20  # kg
    stats['density'] = 100  # kg / m^2
    area = stats['mass'] / stats['density']
    stats['radius'] = np.sqrt(area / np.pi)
    stats['torque'] = 0  # [kg m^2 / s^2] = [ n * m ]
    stats['moment_of_inertia'] = stats['mass'] * stats['radius'] ** 2  # [kg m^2]
    stats['pred'] = True
    return stats

def scale_color(old_color):
    return tuple(0.9 if i > 0.5 else 0 for i in old_color)

if __name__ == "__main__":
    #model = Model([2,4,2])
    #print(model.predict(np.array([[2,2]]).reshape(2,1)))
    print(scale_color((0.8,0,0)))
    from random import randint, random
    for i in range(100):
        print(random())