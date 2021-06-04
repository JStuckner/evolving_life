import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_organism(org, frame, settings):
    radius = int(org.stats['radius'] * 50)
    x = int(round(org.x * settings['resolution'][0] / settings['x_max']))
    y = int(round(org.y * settings['resolution'][1] / settings['y_max']))
    cv2.circle(frame, center=(x, y), radius=radius, color=org.color, thickness = -1)
    cv2.circle(frame, center=(x, y), radius=radius, color=tuple(i/3 for i in org.color), thickness = 2)
    x2 = int(round(np.cos(np.deg2rad(org.r)) * radius * 1.5 + x))
    y2 = int(round(np.sin(np.deg2rad(org.r)) * radius * 1.5 + y))
    cv2.line(frame, (x, y), (x2, y2), color = tuple(i/3 for i in org.color), thickness = 2)
    pass

def plot_food(x1, y1, frame, settings):
    radius = 3
    x = int(round(x1 * settings['resolution'][0] / settings['x_max']))
    y = int(round(y1 * settings['resolution'][1] / settings['y_max']))
    cv2.circle(frame, center=(x, y), radius=radius, color=(255,0,0), thickness = -1)
    pass

def plot_results(gen, best, average, best_pred, average_pred):
    #plt.close()
    plt.clf()
    plt.plot(range(gen), best, label='best prey', color=(0,1,0))
    plt.plot(range(gen), average, label='average prey', color=(0,0.6,0))
    plt.plot(range(gen), best_pred, label='best predator', color='r')
    plt.plot(range(gen), average_pred, label='average predator', color=(1,0.5,0))
    plt.legend()
    plt.xlabel('generation')
    plt.ylabel('number')
    #fig, axs = plt.subplots(2)
    #axs[0].plot(range(gen+1), best)
    #axs[1].plot(range(gen+1), average)
    plt.draw()
    plt.pause(0.0001)

if __name__ == "__main__":
    plot_results(10, range(11), range(11))
    #cv2.circle(np.zeros((10,10)), (1,1))
    #frame = np.zeros((1000, 1000))
    #plot_organism(1,1,1,frame, {'x_max': 4, 'y_max': 4})
    #cv2.imshow('World', frame)
