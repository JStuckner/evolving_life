# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Evolving Simple Organisms
#   2017-Nov.
#
# ------------------------------------------------------------------------------+

# --- IMPORT DEPENDENCIES ------------------------------------------------------+
from __future__ import division, print_function
from collections import defaultdict

import time

import numpy as np
import operator
from joblib import Parallel, delayed
from math import atan2
from math import cos
from math import degrees
from math import floor
from math import radians
from math import sin
from math import sqrt
from random import randint
from random import random
from random import sample
from random import uniform

from plotting import *
from creatures import *

# --- CONSTANTS ----------------------------------------------------------------+

settings = {}

# EVOLUTION SETTINGS
settings['pop_size'] = 60  # number of organisms
settings['pred_pop_size'] = 30 # number of predator organisms
settings['food_num'] = 100  # number of food particles
settings['gens'] = 500  # number of generations
settings['elitism'] = 0.2  # elitism (selection bias)
settings['mutate'] = 0.30  # mutation rate

# SIMULATION SETTINGS
settings['gen_time'] = 100  # generation length         (seconds)
settings['dt'] = 0.06  # simulation time step      (dt)
settings['dr_max'] = 720  # max rotational speed      (degrees per second)
settings['v_max'] = 10  # max velocity              (m /s)
settings['dv_max'] = 1  # max acceleration (+/-)    (units per second^2)
settings['F'] = 0.25  # max acceleration (+/-)    kg m s^-2
settings['x_min'] = 0.  # arena western border
settings['x_max'] = 100.0  # arena eastern border    [m]
settings['y_min'] = 0.  # arena southern border
settings['y_max'] = 100.0  # arena northern border   [m]
settings['dist_to_eat'] = 0.5 # distance to eat food


# ORGANISM NEURAL NET SETTINGS
settings['nodes'] = [6, 6, 6, 2]
settings['layers'] = len(settings['nodes'])


# DISPLAY SETTINGS
settings['plot'] = True  # plot final generation?
settings['resolution'] = [1300 , 1300]

# --- FUNCTIONS ----------------------------------------------------------------+

def dist(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calc_heading(org, obj):
    d_x = obj.x - org.x
    d_y = obj.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r
    if abs(theta_d) > 180: theta_d += 360
    return theta_d / 180


def plot_frame(settings, organisms, foods, gen, time):
    frame = np.full(shape = (settings['resolution'][0],  settings['resolution'][1], 3),
                    fill_value=255,
                    dtype='uint8')

    # PLOT ORGANISMS
    for organism in organisms:
        plot_organism(organism, frame, settings)

    # PLOT FOOD PARTICLES
    for food in foods:
        plot_food(food.x, food.y, frame, settings)

    cv2.imshow('World', frame)
    cv2.waitKey(1)


def evolve(settings, organisms_old, gen):
    elitism_num = int(floor(settings['elitism'] * len(organisms_old)))
    new_orgs = len(organisms_old) - elitism_num

    # --- GET STATS FROM CURRENT GENERATION ----------------+
    stats = defaultdict(int)
    for org in organisms_old:
        if org.fitness > stats['BEST'] or stats['BEST'] == 0:
            stats['BEST'] = org.fitness

        if org.fitness < stats['WORST'] or stats['WORST'] == 0:
            stats['WORST'] = org.fitness

        stats['SUM'] += org.fitness
        stats['COUNT'] += 1

    stats['AVG'] = stats['SUM'] / stats['COUNT']

    # --- ELITISM (KEEP BEST PERFORMING ORGANISMS) ---------+
    orgs_sorted = sorted(organisms_old, key=operator.attrgetter('fitness'), reverse=True)
    organisms_new = []
    new_color = tuple(150 if i > 126 else 0 for i in organisms_old[0].color)
    for i in range(0, elitism_num):
        organisms_new.append(
            Creature(settings, parameters=orgs_sorted[i].brain.parameters, name=orgs_sorted[i].name, color=new_color, stats = organisms_old[i].stats)) #color=tuple(i*0.9 for i in orgs_sorted[i].color)))


    # --- GENERATE NEW ORGANISMS ---------------------------+
    for w in range(0, new_orgs):


        # https://www.ijcai.org/Proceedings/89-1/Papers/122.pdf
        # Training Feedforward Neural Networks Using Genetic Algorithms
        # David Montana and Lawrence Davis

        # SELECTION (TRUNCATION SELECTION)
        canidates = range(0, elitism_num)
        random_index = sample(canidates, 2)
        org_1 = orgs_sorted[random_index[0]]
        org_2 = orgs_sorted[random_index[1]]
        gene_sharing_type = randint(1,2)

        if gene_sharing_type == 1: # CROSSOVER NODES
            weights = []
            biases = []
            for i in range(org_1.brain.n_layers):
                biases.append(org_1.brain.parameters['biases'][i] if randint(1,2) == 1 else org_2.brain.parameters['biases'][i])
                weights.append((np.zeros(org_1.brain.parameters['weights'][i].shape)))
                for col in range(org_1.brain.parameters['weights'][i].shape[1]):
                    weights[-1][:, col] = org_1.brain.parameters['weights'][i][:,col] if randint(1,2) == 1 else org_2.brain.parameters['weights'][i][:,col]

        else:
            # AVERAGE
            crossover_weight = random()
            weights = []
            biases = []
            for i in range(org_1.brain.n_layers):
                weights.append((crossover_weight * org_1.brain.parameters['weights'][i]) + \
                 ((1 - crossover_weight) * org_2.brain.parameters['weights'][i]))
                biases.append((crossover_weight * org_1.brain.parameters['biases'][i]) + \
                 ((1 - crossover_weight) * org_2.brain.parameters['biases'][i]))


        # MUTATION
        # Weight single
        mutate = random()
        if mutate <= settings['mutate']:
            # PICK WHICH WEIGHT MATRIX TO MUTATE
            mat_pick = randint(0, org_1.brain.n_layers-1)
            n_rows, n_cols = weights[mat_pick].shape
            node_pick_x = randint(0, n_rows-1) # Mutate entire node weight
            node_pick_y = randint(0, n_cols-1)
            weights[mat_pick][node_pick_x, node_pick_y] += uniform(-0.2, 0.2) #np.random.uniform(-0.2, 0.2, weights[mat_pick][:, node_pick_y].shape)
        # Weight whole node
        mutate = random()
        if mutate <= settings['mutate']/3:
            # PICK WHICH WEIGHT MATRIX TO MUTATE
            mat_pick = randint(0, org_1.brain.n_layers-1)
            n_rows, n_cols = weights[mat_pick].shape
            node_pick_x = randint(0, n_rows-1) # Mutate entire node weight
            node_pick_y = randint(0, n_cols-1)
            weights[mat_pick][:, node_pick_y] += np.random.uniform(-0.2, 0.2, weights[mat_pick][:, node_pick_y].shape)
        # Bias
        mutate = random()
        if mutate <= settings['mutate']:
            # PICK WHICH WEIGHT MATRIX TO MUTATE
            mat_pick = randint(0, org_1.brain.n_layers-1)
            n_rows = biases[mat_pick].shape[0]
            node_pick_x = randint(0, n_rows-1)
            biases[mat_pick][node_pick_x] += uniform(-0.2, 0.2)



        new_color = tuple(255 if i > 126 else 0 for i in organisms_old[0].color)
        organisms_new.append(
            Creature(settings, parameters={'weights': weights, 'biases': biases}, name='gen[' + str(gen) + ']-org[' + str(w) + ']', color=new_color, stats = organisms_old[i].stats))


    return organisms_new, stats


def simulate(settings, organisms, predators, foods, gen):
    total_time_steps = min(int(settings['gen_time'] / settings['dt']), 3*gen+100)
    #total_time_steps = int(settings['gen_time'] / settings['dt'])
    # --- CYCLE THROUGH EACH TIME STEP ---------------------+
    for t_step in range(0, total_time_steps, 1):

        # PLOT SIMULATION FRAME
        if settings['plot'] == True:  # and gen==settings['gens']-1:
            plot_frame(settings, organisms+predators, foods, gen, t_step)

        # CALCULATE HEADING TO NEAREST FOOD SOURCE AND UPDATE FITNESS
        food_locations = [np.array([f.x, f.y]) for f in foods]
        for org in organisms:
            dists = np.sum((np.array([org.x, org.y]) - food_locations)**2, axis=1)
            closest_food_index = np.argmin(dists)
            closest_dist = np.amin(dists)
            org.d_food = closest_dist
            org.r_food = calc_heading(org, foods[closest_food_index])
            if closest_dist <= settings['dist_to_eat']:
                org.fitness += foods[closest_food_index].energy
                foods[closest_food_index].respawn(settings)

        for org in predators:
            dists = np.sum((np.array([org.x, org.y]) - food_locations)**2, axis=1)
            closest_food_index = np.argmin(dists)
            closest_dist = np.amin(dists)
            org.d_food = closest_dist
            org.r_food = calc_heading(org, foods[closest_food_index])
            if closest_dist <= settings['dist_to_eat'] * 3 and t_step < 2: # respawn food that starts under predators
                foods[closest_food_index].respawn(settings)

        # CALCULATE HEADING TO NEAREST OTHER ORG
        org_locations = [np.array([o.x, o.y]) for o in organisms]
        for org in organisms:
            dists = np.sum((np.array([org.x, org.y]) - org_locations) ** 2, axis=1)
            dists[dists==0] = 9999
            closest_index = np.argmin(dists)
            closest_dist = np.amin(dists)
            org.d_org = closest_dist
            org.r_org = calc_heading(org, organisms[closest_index])

        # CALCULATE HEADING TO NEAREST ORG FROM PRED
        for org in predators:
            dists = np.sum((np.array([org.x, org.y]) - org_locations) ** 2, axis=1)
            dists[dists == 0] = 9999
            closest_index = np.argmin(dists)
            closest_dist = np.amin(dists)
            org.d_org = closest_dist
            org.r_org = calc_heading(org, organisms[closest_index])
            if closest_dist <= settings['dist_to_eat']*1.5:
                org.fitness += 1
                organisms[closest_index].respawn(settings)

        # CALCULATE HEADING TO NEAREST PRED
        pred_locations = [np.array([o.x, o.y]) for o in predators]
        for org in organisms + predators:
            dists = np.sum((np.array([org.x, org.y]) - pred_locations) ** 2, axis=1)
            dists[dists == 0] = 9999
            closest_index = np.argmin(dists)
            closest_dist = np.amin(dists)
            org.d_pred = closest_dist
            org.r_pred = calc_heading(org, predators[closest_index])


        # GET ORGANISM RESPONSE
        for org in organisms + predators:
            org.think()

        # UPDATE ORGANISMS POSITION AND VELOCITY
        for org in organisms + predators:
            org.update_r(settings)
            org.update_vel(settings)
            org.update_pos(settings)

    return organisms, predators


# --- CLASSES ------------------------------------------------------------------+


class food():
    def __init__(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1

    def respawn(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1

# --- MAIN ---------------------------------------------------------------------+


def run(settings):
    # --- POPULATE THE ENVIRONMENT WITH FOOD ---------------+
    foods = []
    for i in range(0, settings['food_num']):
        foods.append(food(settings))

    # --- POPULATE THE ENVIRONMENT WITH ORGANISMS ----------+
    organisms = []
    for i in range(0, settings['pop_size']):
        organisms.append(Creature(settings, name='gen[x]-org[' + str(i) + ']', color=(0,255,0)))

    predators = []
    for i in range(0, settings['pred_pop_size']):
        predators.append(Creature(settings, stats=get_pred_stats(), name='gen[x]-org[' + str(i) + ']', color=(0,0,255)))

    # --- CYCLE THROUGH EACH GENERATION --------------------+
    best = []
    best_pred = []
    average = []
    average_pred = []
    plt.ion()
    #for gen in range(0, settings['gens']):
    gen = 0
    while True:
        gen += 1
        t0 = time.time()

        # SIMULATE
        organisms, predators = simulate(settings, organisms, predators, foods, gen)

        # EVOLVE
        organisms, stats = evolve(settings, organisms, gen)
        predators, pred_stats = evolve(settings, predators, gen)
        #give a predator a prey's brain
        if random() > 0.9:
            predators[-1].brain = organisms[0].brain
            predators[-1].color = (255, 100, 100)

        t = round(time.time() - t0, 2)

        # Show results
        print('> GEN:', gen, 'BEST:', stats['BEST'], 'AVG:', round(stats['AVG'], 2), 'WORST:', stats['WORST'], 'BEST:', pred_stats['BEST'], 'AVG:', round(pred_stats['AVG'], 2), 'WORST:', pred_stats['WORST'], "TIME:", t)
        best.append(stats['BEST'])
        average.append(stats['AVG'])
        best_pred.append(pred_stats['BEST'])
        average_pred.append(pred_stats['AVG'])
        plot_results(gen, best, average, best_pred, average_pred)

    pass


# --- RUN ----------------------------------------------------------------------+

run(settings)

# --- END ----------------------------------------------------------------------+
