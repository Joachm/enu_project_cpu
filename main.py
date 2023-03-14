from LSTM_model import Network, Neurons, Synapses
from rollout import fitness
import gym
#from ES_classes import *
from GA_algorithm import GA
import concurrent.futures
import multiprocessing
import pickle
import numpy as np


import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

gym.logger.set_level(40)

save_every = 5000
EPOCHS = 100 
popsize = 256
cpus = -1
if cpus==-1:
    cpus = multiprocessing.cpu_count()


syn_arch = (3,3,3)
neu_arch = (3,10,3)

init_syn_in = Synapses(1,1,syn_arch)
init_syn_weights_in = init_syn_in.get_params()
len_syn_weights = len(init_syn_weights_in)

init_syn_res1 = Synapses(1,1,syn_arch)
init_syn_weights_res1 = init_syn_res1.get_params()
init_syn_res2 = Synapses(1,1,syn_arch)
init_syn_weights_res2 = init_syn_res2.get_params()


init_syn_out1 = Synapses(1,1,syn_arch)
init_syn_weights_out1 = init_syn_out1.get_params()
init_syn_out2 = Synapses(1,1,syn_arch)
init_syn_weights_out2 = init_syn_out2.get_params()


init_neu = Neurons(1,neu_arch)
init_neu_weights = init_neu.get_params()
len_neu_weights = len(init_neu_weights)
init_neu2 = Neurons(1,neu_arch)
init_neu_weights2 = init_neu2.get_params()


init_weights = np.concatenate((
    init_syn_weights_in, 

    init_syn_weights_res1,
    init_syn_weights_res2,

    init_syn_weights_out1,
    init_syn_weights_out2,

    init_neu_weights, 
    init_neu_weights2, 
    ))


print('number of trainable parameters:', len(init_weights) )


solver = GA(len(init_weights), popsize=popsize, n_elites=16)

'''
solver = OpenES(len(init_weights),
        popsize=popsize,
        rank_fitness=True,
        learning_rate=0.01,
        learning_rate_decay=0.9999,
        learning_rate_limit=0.001,
        sigma_init=0.1,
        sigma_decay=0.999,
        sigma_limit=0.01)

solver.set_mu(init_weights)
#'''



def worker(args):
    
    params, net_size, indx= args

    m = 0
    syn0 = params[m:m+len_syn_weights]
    m += len_syn_weights
    syn1 = params[m:m+len_syn_weights]
    m += len_syn_weights
    syn2 = params[m:m+len_syn_weights]
    m += len_syn_weights
    syn3 = params[m:m+len_syn_weights]
    m += len_syn_weights
    syn4 = params[m:m+len_syn_weights]
    m += len_syn_weights


    neu0 = params[m:m+len_neu_weights]
    m += len_neu_weights
    neu1 = params[m:m+len_neu_weights]
    m += len_neu_weights

    syn_0 = Synapses(net_size[0],net_size[1], syn_arch)
    syn_0.set_params( syn0, syn0, None)

    syn_1 = Synapses(net_size[1],net_size[1], syn_arch)
    syn_1.set_params(syn1, syn2, indx, multiple=True)

    syn_2 = Synapses(net_size[1],net_size[2], syn_arch)
    syn_2.set_params(syn3, syn4, indx, multiple=True)

    neu_0 = Neurons(net_size[1], neu_arch)
    neu_0.set_params(neu0, neu1, indx)

    net = Network(syn_arch, neu_arch, 4,2)
    net.synapses = [syn_0, syn_1, syn_2]
    net.neurons = [neu_0]


    r  = fitness(net)
    return r


pop_mean_curve = np.zeros((EPOCHS))
best_sol_curve = np.zeros((EPOCHS))
evals = []
print('Begin Training')
for epoch in range(EPOCHS):
    
    solutions = solver.ask()
    net_size = (4,20,2)
    res_indx = np.arange(net_size[1])
    np.random.shuffle(res_indx)
    res_indx = res_indx[:net_size[1]//2]    

    with concurrent.futures.ProcessPoolExecutor(cpus) as executor: 
        work_args = [ (sol, net_size, res_indx) for sol in solutions]
        fitlist = executor.map(worker, work_args)
        

    fitlist = np.array(list(fitlist) )
    solver.tell(fitlist)

    pop_mean_curve[epoch] = np.mean(fitlist)
    best_sol_curve[epoch] = np.max(fitlist)
    print('epoch', epoch, 'mean:', pop_mean_curve[epoch], 'best:', best_sol_curve[epoch], 'worst:', np.min(fitlist), 'std:', np.std(fitlist))
    if (epoch+1)%save_every == 0 :

        print('saving')
        print('mean score',np.mean(fitlist))
        print('best score', np.max(fitlist))
        pickle.dump((solver,
            pop_mean_curve,
            best_sol_curve,
            ),open('test_'+str(epoch)+'_'+str(np.mean(fitlist))+ '.pickle', 'wb'))
