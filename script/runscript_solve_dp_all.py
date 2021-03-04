import os
import sys
import numpy as np
import time

import function_board as fb
import function_tool as ft
import function_get_aiming_grid
import function_solve_dp

np.set_printoptions(precision=10)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=300)

#%%
result_dir = '../result/singlegame'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

data_parameter_dir = '../data_parameter/player_gaussin_fit/grid_v2'
data_parameter_dir_fullactionset = '../data_parameter/player_gaussin_fit/grid_full'

playerID_list = [7]

for playerID in playerID_list:
    name_pa = 'player{}'.format(playerID)
    
    print('-------------no turn-------------')
    ## solve the no-turn game with the action set of 984 aiming locations
    print('part grid/ direct method')
    postfix='_partactionset'
    result_filename = result_dir + '/singlegame_{}_noturn{}.pkl'.format(name_pa, postfix)
    [aiming_grid_pa, prob_normalscore_searcharea_pa, prob_singlescore_searcharea_pa, prob_doublescore_searcharea_pa, prob_triplescore_searcharea_pa, prob_bullscore_searcharea_pa] = function_get_aiming_grid.load_aiming_grid(name_pa, data_parameter_dir=data_parameter_dir)
    
    t1 = time.time()
    [optimal_value, optimal_action_index] = function_solve_dp.solve_dp_noturn(prob_normalscore_searcharea_pa, prob_doublescore_searcharea_pa, prob_bullscore_searcharea_pa)
    t2 = time.time()
    print('solve dp_noturn in {} seconds'.format(t2-t1))
    print('optimal_value: {}'.format(optimal_value))
    print('optimal_action_index: {}'.format(optimal_action_index))
    
    result_dic = {'optimal_value':optimal_value, 'optimal_action_index':optimal_action_index}
    ft.dump_pickle(result_filename, result_dic, printflag=True)
    print('\n')

    ## solve the no-turn game with the full action set of 341*341 aiming locations
    print('full grid/ direct method')
    postfix='_fullactionset'
    result_filename = result_dir + '/singlegame_{}_noturn{}.pkl'.format(name_pa, postfix)
    [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = function_get_aiming_grid.load_prob_grid(name_pa, data_parameter_dir=data_parameter_dir_fullactionset)
    
    t1 = time.time()
    [optimal_value_fullactionset, optimal_action_index_fullactionset] = function_solve_dp.solve_dp_noturn(prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    t2 = time.time()
    print('solve dp_noturn with full action set in {} seconds'.format(t2-t1))
    print('optimal_value_fullactionset: {}'.format(optimal_value_fullactionset))
    print('optimal_action_index_fullactionset: {}'.format(optimal_action_index_fullactionset))
    
    result_dic_fullactionset = {'optimal_value':optimal_value_fullactionset, 'optimal_action_index':optimal_action_index_fullactionset}
    ft.dump_pickle(result_filename, result_dic_fullactionset, printflag=True)    
    print('\n')
    
    #document optimal value of noturn
    optimal_value_noturn = optimal_value_fullactionset
    
    ## solve the no-turn game with the action set of 984 aiming locations using value iteration
    print('part grid/ value iteration')
    postfix='_partactionset_valueiter'
    result_filename = result_dir + '/singlegame_{}_noturn{}.pkl'.format(name_pa, postfix)
    [aiming_grid_pa, prob_normalscore_searcharea_pa, prob_singlescore_searcharea_pa, prob_doublescore_searcharea_pa, prob_triplescore_searcharea_pa, prob_bullscore_searcharea_pa] = function_get_aiming_grid.load_aiming_grid(name_pa, data_parameter_dir=data_parameter_dir)
    
    t1 = time.time()
    [optimal_value, optimal_action_index] = function_solve_dp.solve_dp_noturn_valueiteration(prob_normalscore_searcharea_pa, prob_doublescore_searcharea_pa, prob_bullscore_searcharea_pa)
    t2 = time.time()
    print('solve dp_noturn_valueiter in {} seconds'.format(t2-t1))
    print('optimal_value: {}'.format(optimal_value))
    print('optimal_action_index: {}'.format(optimal_action_index))
    
    result_dic = {'optimal_value':optimal_value, 'optimal_action_index':optimal_action_index}
    ft.dump_pickle(result_filename, result_dic, printflag=True)
    print('\n')

    ## solve the no-turn game with the action set of 984 aiming locations using policy iteration
    print('part grid/ policy iteration')
    postfix='_partactionset_policyiter'
    result_filename = result_dir + '/singlegame_{}_noturn{}.pkl'.format(name_pa, postfix)
    [aiming_grid_pa, prob_normalscore_searcharea_pa, prob_singlescore_searcharea_pa, prob_doublescore_searcharea_pa, prob_triplescore_searcharea_pa, prob_bullscore_searcharea_pa] = function_get_aiming_grid.load_aiming_grid(name_pa, data_parameter_dir=data_parameter_dir)
    
    t1 = time.time()
    [optimal_value, optimal_action_index] = function_solve_dp.solve_dp_noturn_policyiteration(prob_normalscore_searcharea_pa, prob_doublescore_searcharea_pa, prob_bullscore_searcharea_pa)
    t2 = time.time()
    print('solve dp_noturn_policyiter in {} seconds'.format(t2-t1))
    print('optimal_value: {}'.format(optimal_value))
    print('optimal_action_index: {}'.format(optimal_action_index))
    
    result_dic = {'optimal_value':optimal_value, 'optimal_action_index':optimal_action_index}
    ft.dump_pickle(result_filename, result_dic, printflag=True)
    print('\n')
    
    print('-------------with turn-------------')
    
    ## solve the turn game with the action set of 984 aiming locations using value iteration
    print('part grid/ value iteration')
    postfix='_partactionset_valueiter'
    result_filename = result_dir + '/singlegame_{}_turn{}.pkl'.format(name_pa, postfix)
    [aiming_grid_pa, prob_normalscore_searcharea_pa, prob_singlescore_searcharea_pa, prob_doublescore_searcharea_pa, prob_triplescore_searcharea_pa, prob_bullscore_searcharea_pa] = function_get_aiming_grid.load_aiming_grid(name_pa, data_parameter_dir=data_parameter_dir)
    
    t1 = time.time()
    result_dic = function_solve_dp.solve_dp_turn_valueiteration(prob_normalscore_searcharea_pa, prob_doublescore_searcharea_pa, prob_bullscore_searcharea_pa, optimal_value_noturn)
    t2 = time.time()
    print('solve dp_turn_policyiter in {} seconds'.format(t2-t1))
    print('optimal_value: {}'.format(result_dic['optimal_value']))
    print('optimal_action_index: {}'.format(result_dic['optimal_action_index']))
    
    #result_dic = {'optimal_value':optimal_value, 'optimal_action_index':optimal_action_index}
    ft.dump_pickle(result_filename, result_dic, printflag=True)
    print('\n')
    

    #run for 83 min!!!!
    ## solve the turn game with the action set of 341*341 aiming locations using value iteration
    print('full grid/ value iteration')
    postfix='_fullactionset_valueiter_turn_value'
    result_filename = result_dir + '/singlegame_{}_noturn{}.pkl'.format(name_pa, postfix)
    [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = function_get_aiming_grid.load_prob_grid(name_pa, data_parameter_dir=data_parameter_dir_fullactionset)
    
    t1 = time.time()
    result_dic = function_solve_dp.solve_dp_turn_valueiteration(prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore, optimal_value_noturn)
    t2 = time.time()
    print('solve dp_noturn_policyiter in {} seconds'.format(t2-t1))
    print('optimal_value: {}'.format(result_dic['optimal_value']))
    print('optimal_action_index: {}'.format(result_dic['optimal_action_index']))
    
    #result_dic = {'optimal_value':optimal_value, 'optimal_action_index':optimal_action_index}
    ft.dump_pickle(result_filename, result_dic, printflag=True)
    print('\n')

    #run for an hour+!!!
    ## solve the turn game with the action set of 984 aiming locations using policy iteration
    print('part grid/ policy iteration')
    postfix='_partactionset_policyiter'
    result_filename = result_dir + '/singlegame_{}_turn{}.pkl'.format(name_pa, postfix)
    [aiming_grid_pa, prob_normalscore_searcharea_pa, prob_singlescore_searcharea_pa, prob_doublescore_searcharea_pa, prob_triplescore_searcharea_pa, prob_bullscore_searcharea_pa] = function_get_aiming_grid.load_aiming_grid(name_pa, data_parameter_dir=data_parameter_dir)
    
    t1 = time.time()
    result_dic = function_solve_dp.solve_dp_turn_policyiteration(prob_normalscore_searcharea_pa, prob_doublescore_searcharea_pa, prob_bullscore_searcharea_pa)
    t2 = time.time()
    print('solve dp_turn_policyiter in {} seconds'.format(t2-t1))
    print('optimal_value: {}'.format(result_dic['optimal_value']))
    print('optimal_action_index: {}'.format(result_dic['optimal_action_index']))
    
    #result_dic = {'optimal_value':optimal_value, 'optimal_action_index':optimal_action_index}
    ft.dump_pickle(result_filename, result_dic, printflag=True)
    print('\n')