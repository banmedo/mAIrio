import os
from re import S

from numpy.lib.function_base import select
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from wrappers import wrapper
# from gym import wrappers

env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = wrappers.Monitor(env, "./gym-results", force=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper(env)

initial_state = env.reset()
n_actions = env.action_space.n

import numpy as np
import random
import keras as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

import logging
log_file = 'log.log'
logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')

def _log(message, level=logging.INFO):
    print(message)
    if level == logging.INFO:
        logging.info(message)
    else:
        logging.error(message)

# default env shape is 240x256x3 but with wrapper that changes
# to 84x84x4
def get_model(input_shape = (84,84,4), actions=n_actions):
    # model = Sequential([
    #     Dense(32, input_shape=input_shape, activation='relu'),
    #     Dropout(0.02),
    #     Dense(16, activation='relu'),
    #     Dropout(0.02),
    #     Flatten(),
    #     Dense(actions, activation = 'softmax')
    # ])
    model = Sequential([
        Conv2D(32, (3,3), input_shape=input_shape, activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(actions, activation='softmax')
    ])
    model.compile()
    return model

def evaluate(model):
    state = env.reset()
    total_reward = 0
    time_wasting_count = 0
    while True:
        action = np.argmax(model.predict(np.array([state])))
        state, reward, done, info = env.step(action)
        total_reward += reward
        # print(f"this step {action} {reward}, {total_reward}, {done}")
        
        if info['life'] < 2:
            done = True
        else:
            if reward == -1:
                time_wasting_count += 1
            elif reward != 0:
                time_wasting_count = 0
            
            if time_wasting_count > 20:
                done = True
                total_reward -= info['time']

        # env.render()
        if done:
            # total_reward += _['score']
            print(f'this run {total_reward}, {done}')
            info['reward'] = reward
            print(info)
            env.reset()
            break
    return total_reward


def selection(models, evaluations, select_num, select_num2=None, gen=0):
    # find out the model ranks based on evaluation
    model_ranks = np.argsort(evaluations)[::-1]
    # sort evaluations
    evaluations = [evaluations[i] for i in model_ranks]
    _log(f'gen{gen} performances')
    _log(evaluations)
    # find and store best unique models
    best_unique = []
    best_unique_evals = []
    rest = []
    for i in model_ranks:
        if len(best_unique) < select_num and evaluations[i] not in best_unique_evals:
            best_unique.append(models[i])
            best_unique_evals.append(evaluations[i])
        else:
            rest.append(models[i])
    # if there arern't enough unique models add good ones from the rest
    more_best_req =  select_num - len(best_unique)
    best_set = best_unique+rest[:more_best_req]
    # archive best model for each gen
    best_set[0].save(f'./cnngamodels/gen{gen}')
    
    if select_num2:
        return best_set, rest[more_best_req:more_best_req+select_num2]
    else:
        return best_set

def unflatten(array, shapes):
    un_array = []
    i = 0
    for shape in shapes:
        size = np.product(shape)
        un_array.append(np.array(array[i:i+size]).reshape(shape))
        i += size
    return un_array


def crossover(models, n_children = 2):
    children = []
    for _ in range(n_children // 2): 
        p1 = random.choice(models)
        p2 = random.choice(models)

        p1_weights = p1.get_weights()
        p2_weights = p2.get_weights()

        shapes = [w.shape for w in p1_weights]

        genes1 = np.concatenate([w.flatten() for w in p1_weights])
        genes2 = np.concatenate([w.flatten() for w in p2_weights])

        split = random.randint(0, len(genes1) - 1)

        child1 = get_model()
        child2 = get_model()

        child1_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
        child2_genes = np.array(genes2[0:split].tolist() + genes1[split:].tolist())

        child1.set_weights(unflatten(child1_genes, shapes))
        child2.set_weights(unflatten(child2_genes, shapes))

        children.append(child1)
        children.append(child2)

    return children
    # models.extend(children)
    # return models


def mutate(model, mutate_ratio=None):
    new_model = k.models.clone_model(model)
    weights = model.get_weights()
    shapes = [w.shape for w in weights]
    flat_weights = np.concatenate([w.flatten() for w in weights])
    for i in range(int(len(flat_weights)*mutate_ratio)) if mutate_ratio else range(random.randint(0, len(flat_weights) // 4)):
        mutate_i = random.randint(0, len(flat_weights) - 1)
    flat_weights[mutate_i] = np.random.randn()
    new_weights = unflatten(flat_weights, shapes)
    new_model.set_weights(new_weights)
    return new_model

def mutate_simple (model, prob=0.25):
    new_model = k.models.clone_model(model)
    if (random.uniform(0.0,1.0) < prob):
        weights = model.get_weights()
        shapes = [w.shape for w in weights]
        flat_weights = np.concatenate([w.flatten() for w in weights])
        mutate_i = random.randint(0, len(flat_weights) - 1)
        flat_weights[mutate_i] = np.random.randn()
        new_weights = unflatten(flat_weights, shapes)
        new_model.set_weights(new_weights)
    return new_model



GENERATIONS = 100 # number of generations to run
POPULATION = 50 # number of networks in a generation
SELECT_BEST = 10 # number of best networks to select
SELECT_SECOND_BEST = 20 # number of lesser networks to preserve (and mutate)
START_WITH_MODEL = './backupmodels/v4.1' # None if starting from scratch

if START_WITH_MODEL:
    base_model = k.models.load_model(START_WITH_MODEL)
    models = [ base_model ] + [ mutate_simple(base_model) for _ in range(POPULATION - 1)]
else:
    models = [ get_model() for _ in range(POPULATION) ]

evaluations = None
for gen in range(0, GENERATIONS):
    print(f"========================  Welcome to gen-{gen}  ==============================")
    # evaluate the models of current gen
    if evaluations:
        evaluations = evaluations[:SELECT_BEST] + [evaluate(model) for model in models[SELECT_BEST:]]
    else:
        evaluations = [evaluate(model) for model in models]
    ##### for just using one best set with limited mutation
    # get the best set of models to breed
    best_set = selection(models, evaluations, SELECT_BEST, gen=gen)
    # breed the best models with crossovers
    breeded_models = crossover(best_set, n_children = POPULATION - len(best_set))
    # possibly mutate the models
    models = best_set + [mutate_simple(model) for model in breeded_models]

    ##### for using best and second best set
    # # get the best set of models to breed
    # best_set, second_best_set = selection(models_sorted, evaluations, SELECT_BEST, SELECT_SECOND_BEST, gen)
    # # breed the best models with crossovers
    # breeded_models = crossover(models, n_children = POPULATION - len(best_set) - len(second_best_set))
    # # possibly mutate the models
    # models = best_set + [mutate(model) for model in breeded_models + second_best_set]

# evaluate the final generation as before
evaluations = [evaluate(model) for model in models]
model_ranks = np.argsort(evaluations)[::-1]
_log(f'gen{GENERATIONS} performances')
_log([evaluations[i] for i in model_ranks])
models[model_ranks[0]].save(f'./cnngamodels/gen{GENERATIONS}')

env.close()
