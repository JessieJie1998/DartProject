import os
import sys
import numpy as np
import time
import sympy

import function_board as fb
import function_tool as ft
#import function_evaluate_policy as fep
import function_get_aiming_grid

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=300)


#%%
## solve the Bellman euqation directly
def solve_dp_noturn(prob_grid_normalscore_searcharea, prob_doublescore_searcharea, prob_bullscore_searcharea):

    ## note that A[..., index] gives the data at the index position from the last dimension of an array A
    
    #possible state: s = 0,1(not possible),2,...,501
    optimal_value = np.zeros(502)
    optimal_value[0] = 0
    optimal_value[1] = np.nan
    optimal_action_index = np.zeros(502, np.int32)
    optimal_action_index[0] = -1
    optimal_action_index[1] = -1
    
    for score_state in range(2,502):            
        ## use matrix operation to search all aiming locations
        
        ## transit to less score state    
        ## s_max = min(score_state-2, 60)
        ## p[z=1]*v[score_state-1] + p[z=2]*v[score_state-2] + ... + p[z=s_max]*v[score_state-s_max]
        score_max = min(score_state-2, 60)
        score_max_plus1 = score_max + 1 
        ## with normal scores 
        num_tothrow = 1.0 + np.dot(prob_grid_normalscore_searcharea[...,1:score_max_plus1], optimal_value[score_state-1:score_state-score_max-1:-1])
        prob_notbust = prob_grid_normalscore_searcharea[...,1:score_max_plus1].sum(axis=-1)
        
        ## transit to finishing
        if (score_state == fb.score_DB): ## hit double bull
            prob_notbust += prob_bullscore_searcharea[...,1]
        elif (score_state <= 40 and score_state%2==0): ## hit double
            doublescore_index = (score_state/2) - 1
            prob_notbust += prob_doublescore_searcharea[...,doublescore_index]
        else: ## not able to finish
            pass
        
        ## expected number of throw for all 341*341 aiming locations
        prob_notbust = np.maximum(prob_notbust, 0)
        tempvalue = num_tothrow / (prob_notbust+1e-18)
                            
        ## searching
        optimal_value[score_state] = np.min(tempvalue)
        optimal_action_index[score_state] = np.argmin(tempvalue)

    return [optimal_value, optimal_action_index]

#%%  project to do 
## solve the no-turn game Bellman euqation using value iteration
def solve_dp_noturn_valueiteration(prob_grid_normalscore_searcharea, prob_doublescore_searcharea, prob_bullscore_searcharea):
   
    #possible state: s = 0,1(not possible),2,...,501
    optimal_value = np.zeros(502) + 15
    
    optimal_value[0] = 0
    optimal_value[1] = np.nan
    optimal_action_index = np.zeros(502, np.int32)
    optimal_action_index[0] = -1
    optimal_action_index[1] = -1
    
    ## implement the value iteration method
    #iteration_relerror_limit = 10**-3
    iteration_relerror_limit = 10**-3
 
        
    #value iteration in backward induction
    for score_state in range(2,502):
        
        score_max = min(score_state-2, 60)
        score_max_plus1 = score_max + 1  
        
        prob_notbust = prob_grid_normalscore_searcharea[...,1:score_max_plus1].sum(axis=-1) 
        
        ## transit to finishing
        if (score_state == fb.score_DB): ## hit double bull
            prob_notbust += prob_bullscore_searcharea[...,1]
        elif (score_state <= 40 and score_state%2==0): ## hit double
            doublescore_index = (score_state/2) - 1
            prob_notbust += prob_doublescore_searcharea[...,doublescore_index]
        else: ## not able to finish
            pass
        prob_notbust = np.maximum(prob_notbust, 0)
        
        error = 1
        
        while error > iteration_relerror_limit:
        
            previous_value = optimal_value[score_state]
            
            ## p[z=1]*v[score_state-1] + p[z=2]*v[score_state-2] + ... + p[z=s_max]*v[score_state-s_max]
            part1 = 1.0 + np.dot(prob_grid_normalscore_searcharea[...,1:score_max_plus1], optimal_value[score_state-1:score_state-score_max-1:-1])
            #bust or score 0
            part2 = (1-prob_notbust)*optimal_value[score_state]
            
            tempvalue = part1 + part2
            
            optimal_value[score_state] = np.min(tempvalue)
            optimal_action_index[score_state] = np.argmin(tempvalue)
            error = abs(optimal_value[score_state] - previous_value)
        
    
    return [optimal_value, optimal_action_index]

def solve_dp_noturn_policyiteration(prob_grid_normalscore_searcharea, prob_doublescore_searcharea, prob_bullscore_searcharea):
   
    prob_grid_normalscore_searcharea = prob_grid_normalscore_searcharea.reshape(-1,61)
    prob_doublescore_searcharea = prob_doublescore_searcharea.reshape(-1,20)
    prob_bullscore_searcharea = prob_bullscore_searcharea.reshape(-1,2)
    
    #possible state: s = 0,1(not possible),2,...,501
    optimal_value = np.zeros(502)
    optimal_value[0] = 0
    optimal_value[1] = np.nan
    optimal_action_index = np.zeros(502, np.int32)
    optimal_action_index[0] = -1
    optimal_action_index[1] = -1
    
    ## implement the policy iteration method
    iteration_relerror_limit = 10**-8

    #policy iteration in backward induction
    for score_state in range(2,502):
        
        score_max = min(score_state-2, 60)
        score_max_plus1 = score_max + 1  
        
        #denominator shape = (984,)
        prob_notbust = prob_grid_normalscore_searcharea[...,1:score_max_plus1].sum(axis=-1) 
        
        ## transit to finishing
        if (score_state == fb.score_DB): ## hit double bull
            prob_notbust += prob_bullscore_searcharea[...,1]
        elif (score_state <= 40 and score_state%2==0): ## hit double
            doublescore_index = (score_state/2) - 1
            prob_notbust += prob_doublescore_searcharea[...,doublescore_index]
        else: ## not able to finish
            pass
        
        ## expected number of throw for all 341*341 aiming locations
        prob_notbust = np.maximum(prob_notbust, 0)
        
        error = 1
        while error > iteration_relerror_limit:
            
            #solve initial policy
            previous_policy = optimal_action_index[score_state]
            numerator = 1.0 + np.dot(prob_grid_normalscore_searcharea[previous_policy,1:score_max_plus1], optimal_value[score_state-1:score_state-score_max-1:-1])
            optimal_value[score_state] = numerator/(prob_notbust[previous_policy] + 1e-18) #prevent zero division
            
            #policy iteration (341,341)
            tempvalue = 1.0 + np.dot(prob_grid_normalscore_searcharea[...,1:score_max_plus1], optimal_value[score_state-1:score_state-score_max-1:-1]) + (1-prob_notbust)*optimal_value[score_state]              
            optimal_action_index[score_state] = np.argmin(tempvalue)
            error = abs(optimal_action_index[score_state] - previous_policy)
    
    
    return [optimal_value, optimal_action_index]


## solve the turn game Bellman euqation using value iteration
def solve_dp_turn_valueiteration(prob_grid_normalscore_searcharea, prob_doublescore_searcharea, prob_bullscore_searcharea, optimal_value_noturn=None, init_num=0, div_num=2):
 
    prob_grid_normalscore_searcharea = prob_grid_normalscore_searcharea.reshape(-1,61)
    prob_doublescore_searcharea = prob_doublescore_searcharea.reshape(-1,20)
    prob_bullscore_searcharea = prob_bullscore_searcharea.reshape(-1,2)
    #possible state: s = 0,1(not possible),2,...,501
    
    optimal_value = np.zeros((502,3,121))+init_num
    optimal_action_index = np.zeros((502,3,121),np.int32)
    
    #initialize
    for i in range(3):
        for u in range(121):
            
            optimal_value[1,i,u] = np.nan
            optimal_action_index[1,i,u] = -1
            
            optimal_action_index[0,i,u] = -1
            optimal_value[0,i,u] = 0
    
    
    #initialize by noturn value
    if optimal_value_noturn is not None:
        print('initialize by optimal_value_noturn')
        for s in range(502):
            for i in range(3):
                for u in range(121):
                    if s < u:
                        optimal_value[s,i,u] = np.nan
                        optimal_action_index[s,i,u] = 0
                    elif not np.isnan(optimal_value[s,i,u]):
                        optimal_value[s,i,u] = optimal_value_noturn[s-u]/div_num
                  
    
    
    
    iteration_relerror_limit = 1e-8
    #value iteration in backward induction
    for score_state in range(2,502):
        error = 1
        while error >  iteration_relerror_limit:
            optimal_value_prev = optimal_value.copy()
        #iter 183 equations
            for i in [2,1,0]:
                if i == 2:### 3 rounds left
                    u = 0
                    h_max = min(score_state-2, 60)
                    h_max_plus1 = h_max + 1
                    #print(h_max_plus1 )
    
                    #p(h)*V(s,2,h)
                    part1 = np.dot(prob_grid_normalscore_searcharea[...,0:h_max_plus1], optimal_value[score_state,1,0:h_max_plus1])
                    #
                    #calculate bust =1-notbust-0
                    part2 = np.zeros(part1.shape[0])
                    prob_notbust = prob_grid_normalscore_searcharea[...,0:h_max_plus1].sum(axis=-1)
                    if (score_state == fb.score_DB): ## hit double bull
                        prob_notbust += prob_bullscore_searcharea[...,1]
                        part2 += prob_bullscore_searcharea[...,1]
                    elif (score_state <= 40 and score_state%2==0): ## hit double
                        doublescore_index = int(score_state/2) - 1
                        prob_notbust += prob_doublescore_searcharea[...,doublescore_index]
                        part2 += prob_doublescore_searcharea[...,int(score_state/2)-1]
                    else: ## not able to finish
                        pass
                    prob_notbust = np.maximum(prob_notbust, 0)
                    prob_bust = 1 - prob_notbust
                    prob_bust = np.maximum(prob_bust, 0)
                    
                    part3 = (optimal_value[score_state,2,0]+1)*prob_bust
    
                    tempvalue = part1 + part2 + part3
    
                    
                    optimal_value[score_state,i,u] = np.min(tempvalue)
                    optimal_action_index[score_state,i,u] = np.argmin(tempvalue)

    
                if i == 1:###two rounds left
                    for u in range(min(61,score_state-1)):
                        
                        h_max = min(score_state-2-u, 60)
                        h_max_plus1 = h_max + 1
     
                        part1 = np.dot(prob_grid_normalscore_searcharea[...,0:h_max_plus1], optimal_value[score_state,0,u:(h_max_plus1+u)])
                        #bust or score 0
                        
                        part2 = np.zeros(part1.shape[0])
    
                        prob_notbust = prob_grid_normalscore_searcharea[...,0:h_max_plus1].sum(axis=-1)
                        if (score_state-u == fb.score_DB): ## hit double bull
                            part2 += prob_bullscore_searcharea[...,1]
                            prob_notbust += prob_bullscore_searcharea[...,1]
                        elif (score_state-u <= 40 and (score_state-u)%2==0): ## hit double
                            doublescore_index = int((score_state-u)/2) - 1
                            part2 += prob_doublescore_searcharea[...,doublescore_index]
                            prob_notbust += prob_doublescore_searcharea[...,doublescore_index]
                        else: ## not able to finish
                            pass
                        prob_notbust = np.maximum(prob_notbust, 0)
                        
                        prob_bust = 1 - prob_notbust              
                        prob_bust = np.maximum(prob_bust, 0)
                        
                        part3 = prob_bust*(optimal_value[score_state,2,0]+1)
                        tempvalue = part1 + part2 + part3
                        
                        optimal_value[score_state,i,u] = np.min(tempvalue)
                        optimal_action_index[score_state,i,u] = np.argmin(tempvalue)
    
                        
                if i == 0:###one rounds left
                    for u in range(min(121,score_state-1)):
    
                
                        h_max = min(score_state-2-u, 60)
                        h_max_plus1 = h_max + 1 
                        
                        part1 = np.dot(prob_grid_normalscore_searcharea[...,0:h_max_plus1], 1+optimal_value[(score_state-u):(score_state-u-h_max_plus1):-1,2,0])                    
                        
                        part2 = np.zeros(part1.shape[0])
                        prob_notbust = prob_grid_normalscore_searcharea[...,0:h_max_plus1].sum(axis=-1)
                        if (score_state-u == fb.score_DB): ## hit double bull
                            part2 += prob_bullscore_searcharea[...,1]
                            prob_notbust += prob_bullscore_searcharea[...,1]
                        elif (score_state-u <= 40 and (score_state-u)%2==0): ## hit double
                            doublescore_index = int((score_state-u)/2) - 1
                            part2 += prob_doublescore_searcharea[...,doublescore_index]
                            prob_notbust += prob_doublescore_searcharea[...,doublescore_index]
                        else: ## not able to finish
                            pass
                        prob_notbust = np.maximum(prob_notbust, 0)
                        prob_bust = 1 - prob_notbust
                        prob_bust = np.maximum(prob_bust, 0)
                        #bust or score 0
                        part3 = prob_bust*(optimal_value[score_state,2,0]+1)
                        
                        tempvalue = part1+ part2+ part3
                        
                        optimal_value[score_state,i,u] = np.min(tempvalue)
                        optimal_action_index[score_state,i,u] = np.argmin(tempvalue)
                        
                mdat = np.ma.masked_array(optimal_value_prev[score_state] - optimal_value[score_state],np.isnan(optimal_value_prev[score_state] - optimal_value[score_state]))
                error = abs(mdat).sum()
                #error = abs(optimal_value_prev[score_state] - optimal_value[score_state]).sum()
                
        #break

    result_dic = {'optimal_value':optimal_value, 'optimal_action_index':optimal_action_index}
    return result_dic

## solve the turn game Bellman euqation using value iteration
## this is optional     
def solve_equation_speedup(equations,vrs1,vrs2,vrs3):
    A  = []
    allb = []
    for equa in equations:
        cof = []
        b = -equa
        cof.append(equa.coeff(vrs3))
        b += equa.coeff(vrs3)*vrs3
        for var in vrs2:
            cof.append(equa.coeff(var))
            b += equa.coeff(var)*var
        for var in vrs1:
            cof.append(equa.coeff(var))
            b += equa.coeff(var)*var  
        allb.append(b)
        A.append(cof)
    
    ans = np.linalg.solve(np.array(A,dtype='float'),np.array(allb,dtype='float'))
    v3 = ans[0]
    v2 = ans[1:(len(vrs2)+1)]
    v1 = ans[(len(vrs2)+1):]
    return [v3,v2,v1]

def solve_dp_turn_policyiteration(prob_grid_normalscore_searcharea, prob_doublescore_searcharea, prob_bullscore_searcharea):

     
    prob_grid_normalscore_searcharea = prob_grid_normalscore_searcharea.reshape(-1,61)
    prob_doublescore_searcharea = prob_doublescore_searcharea.reshape(-1,20)
    prob_bullscore_searcharea = prob_bullscore_searcharea.reshape(-1,2)
    
    #possible state: s = 0,1(not possible),2,...,501
    ## implement the policy iteration method
    optimal_value = np.zeros((502,3,121),dtype=object)
    optimal_action_index = np.zeros((502,3,121),np.int32)+941
        #initialize
    for i in range(3):
        for u in range(121):
            
            optimal_value[1,i,u] = np.nan
            optimal_action_index[1,i,u] = -1
            
            optimal_action_index[0,i,u] = -1#np.nan
            optimal_value[0,i,u] = 0
    
    for s in range(502):
        for i in range(3):
            for u in range(s,121):
                optimal_action_index[0,i,u] = -1
    
    
    
    #value iteration in backward induction
    for score_state in range(2,502):
        print(score_state)
        error = 1
        while error > 1e-4:
            value_prev = optimal_value.copy()
            #using previous infomation solve the equation of V(s,i,u)
            equations, variables = [], []
        
            #define  variables
            #i = 2
            vrs3 =  sympy.symbols('viii')
            optimal_value[score_state,2,0] = vrs3
            variables.append(vrs3)
            #i = 1
            num_var2 = min(60,score_state-2)+1
            vrs2 =  sympy.symbols('vii:'+str(num_var2))
            optimal_value[score_state,1,:num_var2] = vrs2
            variables += list(vrs2)
            #i = 0
            num_var1 = min(120,score_state-2)+1
            vrs1 =  sympy.symbols('vi:'+str(num_var1))
            optimal_value[score_state,0,:num_var1] = vrs1[:num_var1]
            variables += list(vrs1)
            
            i, u = 2, 0
            h_max = min(score_state-2, 60)
            h_max_plus1 = h_max + 1
            policy = optimal_action_index[score_state,i,0]
            part1 = np.dot(prob_grid_normalscore_searcharea[policy,0:h_max_plus1], optimal_value[score_state,1,0:h_max_plus1])
            
            part2 = 0
            prob_notbust = prob_grid_normalscore_searcharea[policy,0:h_max_plus1].sum(axis=-1)
            if (score_state == fb.score_DB): ## hit double bull
                prob_notbust += prob_bullscore_searcharea[policy,1]
                part2 += prob_bullscore_searcharea[policy,1]
            elif (score_state <= 40 and score_state%2==0): ## hit double
                doublescore_index = int(score_state/2) - 1
                prob_notbust += prob_doublescore_searcharea[policy,doublescore_index]
                part2 += prob_doublescore_searcharea[policy,int(score_state/2)-1]
            else: ## not able to finish
                pass
            prob_notbust = np.maximum(prob_notbust, 0)
            prob_bust = 1 - prob_notbust
            prob_bust = np.maximum(prob_bust, 0)
            
            part3 = (optimal_value[score_state,i,u]+1)*prob_bust
            
            tempvalue = part1 + part2 + part3
        
            equations.append(tempvalue-vrs3)
    
            i = 1
            for u in range(min(61,score_state-1)):
                policy = optimal_action_index[score_state, i, u]
                h_max = min(score_state-2-u, 60)
                h_max_plus1 = h_max + 1
         
                part1 = np.dot(prob_grid_normalscore_searcharea[policy,0:h_max_plus1], optimal_value[score_state,0,u:(h_max_plus1+u)])
                #bust or score 0
                
                part2 = 0
        
                prob_notbust = prob_grid_normalscore_searcharea[policy,0:h_max_plus1].sum(axis=-1)
                if (score_state-u == fb.score_DB): ## hit double bull
                    part2 += prob_bullscore_searcharea[policy,1]
                    prob_notbust += prob_bullscore_searcharea[policy,1]
                elif (score_state-u <= 40 and (score_state-u)%2==0): ## hit double
                    doublescore_index = int((score_state-u)/2) - 1
                    part2 += prob_doublescore_searcharea[policy,doublescore_index]
                    prob_notbust += prob_doublescore_searcharea[policy,doublescore_index]
                else: ## not able to finish
                    pass
                prob_notbust = np.maximum(prob_notbust, 0)
                
                prob_bust = 1 - prob_notbust              
                prob_bust = np.maximum(prob_bust, 0)
                
                part3 = prob_bust*(optimal_value[score_state,2,0]+1)
                
                tempvalue = part1 + part2 + part3
                
                equations.append(tempvalue-vrs2[u])
                
            i = 0
            for u in range(min(121,score_state-1)):
                policy = optimal_action_index[score_state, i, u]
                h_max = min(score_state-2-u, 60)
                h_max_plus1 = h_max + 1 
                
                part1 = np.dot(prob_grid_normalscore_searcharea[policy,0:h_max_plus1], 1+optimal_value[(score_state-u):(score_state-u-h_max_plus1):-1,2,0])                    
                
                part2 = 0
                prob_notbust = prob_grid_normalscore_searcharea[policy,0:h_max_plus1].sum(axis=-1)
                if (score_state-u == fb.score_DB): ## hit double bull
                    part2 += prob_bullscore_searcharea[policy,1]
                    prob_notbust += prob_bullscore_searcharea[policy,1]
                elif (score_state-u <= 40 and (score_state-u)%2==0): ## hit double
                    doublescore_index = int((score_state-u)/2) - 1
                    part2 += prob_doublescore_searcharea[policy,doublescore_index]
                    prob_notbust += prob_doublescore_searcharea[policy,doublescore_index]
                else: ## not able to finish
                    pass
                prob_notbust = np.maximum(prob_notbust, 0)
                prob_bust = 1 - prob_notbust
                prob_bust = np.maximum(prob_bust, 0)
                #bust or score 0
                part3 = prob_bust*(optimal_value[score_state,2,0]+1)
                
                tempvalue = part1+ part2+ part3
                
                equations.append(tempvalue-vrs1[u])
        
            #ans = sympy.solve(equations,variables)
            v3, v2, v1 = solve_equation_speedup(equations,vrs1,vrs2,vrs3)
            
            #Update  variables
            #i = 2  
            optimal_value[score_state,2,0] = v3
        
            #i = 1
            for u in range(0,num_var2):
                optimal_value[score_state,1,u] =  v2[u]
        
            #i = 0
            for u in range(0,num_var1):
                optimal_value[score_state,0,u] =  v1[u]
              
            
            for i in [2,1,0]:
                if i == 2:### 3 rounds left
                    u = 0
                    h_max = min(score_state-2, 60)
                    h_max_plus1 = h_max + 1
                    #print(h_max_plus1 )
            
                    #p(h)*V(s,2,h)
                    part1 = np.dot(prob_grid_normalscore_searcharea[...,0:h_max_plus1], optimal_value[score_state,1,0:h_max_plus1])
                    #
                    #calculate bust =1-notbust-0
                    part2 = np.zeros(part1.shape[0])
                    prob_notbust = prob_grid_normalscore_searcharea[...,0:h_max_plus1].sum(axis=-1)
                    if (score_state == fb.score_DB): ## hit double bull
                        prob_notbust += prob_bullscore_searcharea[...,1]
                        part2 += prob_bullscore_searcharea[...,1]
                    elif (score_state <= 40 and score_state%2==0): ## hit double
                        doublescore_index = int(score_state/2) - 1
                        prob_notbust += prob_doublescore_searcharea[...,doublescore_index]
                        part2 += prob_doublescore_searcharea[...,int(score_state/2)-1]
                    else: ## not able to finish
                        pass
                    prob_notbust = np.maximum(prob_notbust, 0)
                    prob_bust = 1 - prob_notbust
                    prob_bust = np.maximum(prob_bust, 0)
                    
                    part3 = (optimal_value[score_state,2,0]+1)*prob_bust
            
                    tempvalue = part1 + part2 + part3
            
                    
                    optimal_action_index[score_state,i,u] = np.argmin(tempvalue)
            
            
                if i == 1:###two rounds left
                    for u in range(min(61,score_state-1)):
                        
                        h_max = min(score_state-2-u, 60)
                        h_max_plus1 = h_max + 1
             
                        part1 = np.dot(prob_grid_normalscore_searcharea[...,0:h_max_plus1], optimal_value[score_state,0,u:(h_max_plus1+u)])
                        #bust or score 0
                        
                        part2 = np.zeros(part1.shape[0])
            
                        prob_notbust = prob_grid_normalscore_searcharea[...,0:h_max_plus1].sum(axis=-1)
                        if (score_state-u == fb.score_DB): ## hit double bull
                            part2 += prob_bullscore_searcharea[...,1]
                            prob_notbust += prob_bullscore_searcharea[...,1]
                        elif (score_state-u <= 40 and (score_state-u)%2==0): ## hit double
                            doublescore_index = int((score_state-u)/2) - 1
                            part2 += prob_doublescore_searcharea[...,doublescore_index]
                            prob_notbust += prob_doublescore_searcharea[...,doublescore_index]
                        else: ## not able to finish
                            pass
                        prob_notbust = np.maximum(prob_notbust, 0)
                        
                        prob_bust = 1 - prob_notbust              
                        prob_bust = np.maximum(prob_bust, 0)
                        
                        part3 = prob_bust*(optimal_value[score_state,2,0]+1)
                        tempvalue = part1 + part2 + part3
                        
                        optimal_action_index[score_state,i,u] = np.argmin(tempvalue)
            
                        
                if i == 0:###one rounds left
                    for u in range(min(121,score_state-1)):
            
                
                        h_max = min(score_state-2-u, 60)
                        h_max_plus1 = h_max + 1 
                        
                        part1 = np.dot(prob_grid_normalscore_searcharea[...,0:h_max_plus1], 1+optimal_value[(score_state-u):(score_state-u-h_max_plus1):-1,2,0])                    
                        
                        part2 = np.zeros(part1.shape[0])
                        prob_notbust = prob_grid_normalscore_searcharea[...,0:h_max_plus1].sum(axis=-1)
                        if (score_state-u == fb.score_DB): ## hit double bull
                            part2 += prob_bullscore_searcharea[...,1]
                            prob_notbust += prob_bullscore_searcharea[...,1]
                        elif (score_state-u <= 40 and (score_state-u)%2==0): ## hit double
                            doublescore_index = int((score_state-u)/2) - 1
                            part2 += prob_doublescore_searcharea[...,doublescore_index]
                            prob_notbust += prob_doublescore_searcharea[...,doublescore_index]
                        else: ## not able to finish
                            pass
                        prob_notbust = np.maximum(prob_notbust, 0)
                        prob_bust = 1 - prob_notbust
                        prob_bust = np.maximum(prob_bust, 0)
                        #bust or score 0
                        part3 = prob_bust*(optimal_value[score_state,2,0]+1)
                        
                        tempvalue = part1+ part2+ part3                   
                        optimal_action_index[score_state,i,u] = np.argmin(tempvalue)
            
            error = np.nansum(np.abs(value_prev - optimal_value))
            #print(score_state,optimal_value[score_state,2,0])
    result_dic = {'optimal_value':optimal_value, 'optimal_action_index':optimal_action_index}
    return result_dic






