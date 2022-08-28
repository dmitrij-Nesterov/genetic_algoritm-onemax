import random
import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools, algorithms

MAX_LENGHT = 100

POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATION = 50

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

def oneMaxFitness(individual):
    return sum(individual),

toolbox = base.Toolbox()
toolbox.register('zero_or_one', random.randint, 0, 1)
toolbox.register('individual_creator', tools.initRepeat, creator.Individual, toolbox.zero_or_one, MAX_LENGHT)
toolbox.register('population_creator', tools.initRepeat, list, toolbox.individual_creator)

population = toolbox.population_creator(n=POPULATION_SIZE)

toolbox.register('evaluate', oneMaxFitness)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=1.0/MAX_LENGHT)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('values', list)

population, logbook = algorithms.eaSimple(population, toolbox,
                                          cxpb=P_CROSSOVER,
                                          mutpb=P_MUTATION,
                                          ngen=MAX_GENERATION,
                                          stats=stats,
                                          verbose=True)

vals = logbook.select('values')

import time

plt.ion()
fig, ax = plt.subplots()

line, = ax.plot(vals[0], ' o', markersize=1)
ax.set_ylim(-10, 110)

for v in vals:
    line.set_ydata(v)

    plt.draw()
    plt.gcf().canvas.flush_events()

    time.sleep(0.5)

plt.ioff()
plt.show()