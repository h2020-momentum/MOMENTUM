
import pandas as pd 
import numpy as np 
import math
from collections import defaultdict
from typing import List, Tuple
from random import seed, randint
from itertools import product, combinations
from math import sqrt
import networkx as nx
from mip import Model, xsum, BINARY, minimize, ConstrsGenerator, CutPool, INTEGER
import time
import matplotlib.pyplot as plt


# n = 24  # number of points
# V = set(range(n))
# places =  [ str(i) for i in V]
# seed(0)
# p = [(randint(1, 100), randint(1, 100)) for i in places]  # coordinates
# # # Arcs = [(i, j) for (i, j) in product(V, V) if i != j]

# # # distance matrix
# overall_distances= [[round(sqrt((p[i][0]-p[j][0])**2 + (p[i][1]-p[j][1])**2)) for j in V] for i in V]

t0 = time.time()

overall_distances = [[0, 5.2, 1.6, 3, 6.2, 6.8, 4.5, 1.6],
            [5.2, 0, 5, 1.8, 2.3, 3.2, 2, 3.6],
            [1.6, 5, 0, 4.2, 6, 6.5, 4.1, 2],
            [3, 1.8, 4.2, 0, 3.2, 4.2, 1.5, 2],
            [6.2, 2.3, 6, 3.2, 0, 2.2, 3.1, 4.6],
            [6.8, 3.2, 6.5, 4.2, 2.2, 0, 4.2, 6.2],
            [4.5, 2, 4.1, 1.5, 3.1, 4.2, 0, 3.2],
            [1.6, 3.6, 2, 2, 4.6, 6.2, 3.2, 0]]
            
            
invertory = [31, 10, 7,  3,  6, 0,  0,   0]
capacity = [100, 50, 50, 12, 50, 10, 12, 14]
predicted_demand = [4, 1, 0, 5, 0, 7, 0, 1]
# # predicted_demand = [4, 1, 0, 0, 0, 0, 0, 1]
# # predicted_demand = [4, 1, 0, 4, 5, 2, 0, 1]

            
# invertory = [30, 22, 5,  3,  35, 5,  6,   8]
# capacity = [100, 50, 50, 12, 50, 10, 12, 14]
# predicted_demand = [30, 23, 14, 10, 26, 0, 0, 11]

# capacity = [np.random.randint(low = 30, high = 50) for _ in range(n)]
# invertory = [ cap - np.random.randint(low = 15, high = 40) for i, cap in enumerate(capacity)]
# predicted_demand =  [ inv -  np.random.randint(low = -15, high = 30 ) for i, inv in enumerate(invertory)]
            
# invertory = [31, 10, 7,  3,  6, 0,  0,   0, 31, 10, 7,  3,  6, 0,  0,   0, 31, 10, 7,  3,  6, 0,  0,   0 ]
# capacity = [100, 50, 50, 12, 50, 10, 12, 14, 100, 50, 50, 12, 50, 10, 12, 14, 100, 50, 50, 12, 50, 10, 12, 14]
# predicted_demand = [4, 1, 0, 5, 0, 7, 0, 1, 4, 1, 0, 5, 0, 7, 0, 1, 4, 1, 0, 5, 0, 7, 0, 1]
# print(len(invertory))
# print(len(capacity))
# print(len(predicted_demand))

real_demand = []
for i in range(len(invertory)):
    rd = predicted_demand[i] - invertory[i]
    if rd == 0:
        real_demand.append(0)
    elif rd > 0:
        x = min(rd,capacity[i] - invertory[i])
        real_demand.append(x) 
    else:
        x= min(abs(rd),invertory[i])
        real_demand.append(-x)

# real_demand = []

places = ['loc1', 'loc2', 'loc3', 'loc4', 'loc5', 'loc6', 'loc7', 'loc8']

demand = {place: rde for place,rde in zip(places, real_demand)}
print(demand)


# The cost data is made into a dictionary
cost = {p :{ p2 : overall_distances[i][j] for j,  p2 in enumerate(places) } for j, p in enumerate(places)}

m = Model()

# Creates a list of tuples containing all the possible routes for transport
Routes = [(w,b) for w in places for b in places if w !=b]

# A dictionary called 'Vars' is created to contain the referenced variables(the routes)
x = {n: { a2 : m.add_var(name="Route_{}_{}".format(n, a2), var_type=INTEGER, lb= 0, ub=2) for a2 in places if a2 != n} for n in places}

# vars_2 = LpVariable.dicts('Flows', (places,places),0,None,LpInteger)
f = {n: { a2 : m.add_var(name="Flows_{}_{}".format(n, a2)) for a2 in places if a2 != n} for n in places}


# The objective function is added to 'prob' first

m.objective = xsum(cost[o][d] * x[o][d] for (o, d) in Routes)


for i in places:
   
    m += xsum([x[p][i] for p in places if i!=p]) == xsum([x[i][p] for p in places  if i!=p])
    
        
#must start from loc1.... 
m += xsum([x[places[0]][p] for p in places[1:]]) >=1

Q = 8

#flow constraints-----------
for j in places:
    if demand[j] > 0: 
        m += xsum([f[j][i] for i in places if i!=j]) - xsum([f[i][j] for i in places if i!=j]) == - demand[j]
    else:
        m += xsum([f[j][i] for i in places if j!=i]) - xsum([f[i][j] for i in places if i!=j]) <= - demand[j]

for i in places[1:]:
    for g in places[1:]:
        if i != g:
            rest =  [k for k in places if (k != i and k != g)]
            m += x[i][g] <= xsum([x[j][i] for j in rest ])
            

class SubTourCutGenerator(ConstrsGenerator):
    """Class to generate cutting planes for the TSP"""
    def __init__(self, Fl: List[Tuple[int, int]], x_, V_):
        self.F, self.x, self.V = Fl, x_, V_

    def generate_constrs(self, model: Model, depth: int = 0, npass: int = 0):
        xf, V_, cp, G = model.translate(self.x), self.V, CutPool(), nx.DiGraph()
        for (u, v) in [(k, l) for (k, l) in product(V_, V_) if k != l and xf[k][l]]:
            G.add_edge(u, v, capacity=xf[u][v].x)
        for (u, v) in F:
            val, (S, NS) = nx.minimum_cut(G, u, v)
            if val <= 0.99:
                aInS = [(xf[i][j], xf[i][j].x)
                        for (i, j) in product(V_, V_) if i != j and xf[i][j] and i in S and j in S]
                if sum(f for v, f in aInS) >= (len(S) + sum([1 if demand[i]  > Q else 0 for i in S ]) -1)+1e-4  + xsum([x[g][l] for g in NS for l in S if (g!=l)]):
                    cut = xsum(1.0*v for v, fm in aInS) <= len(S)-1 + sum([1 if demand[i] > Q else 0 for i in S ]) + xsum([x[g][l] for g in NS for l in S if (g!=l)])
                    cp.add(cut)
                    if len(cp.cuts) > 256:
                        for cut in cp.cuts:
                            model += cut
                        return
        for cut in cp.cuts:
            model += cut

F, G = [], nx.DiGraph()
for (i, j) in Routes:
    G.add_edge(i, j, weight=cost[i][j])

# nx.draw(G, with_labels = True)
for i in places:
    P, D = nx.dijkstra_predecessor_and_distance(G, source=i)
    # print(P, D)
    DS = list(D.items())
    # print('\n', DS)
    DS.sort(key=lambda x: x[1])
    # print('\n', DS, '\n')
    F.append((i, DS[-1][0]))

m.cuts_generator = SubTourCutGenerator(F, x, places)

for i in places:
    for j in places: 
        # prob += vars_2[i][j] >= max(0,-demand[i],demand[j])*vars[i][j]
        if i != j :
            m += f[i][j] >= 0

            
        
for i in places:
    for j in places: 
        # prob += vars_2[i][j] <= min(Q,Q-demand[i],Q + demand[j])*vars[i][j]
        if i != j :
            m += f[i][j] <= Q*x[i][j]


# The problem is solved using PuLP's choice of Solver
m.optimize()

t1 = time.time()
print(t1-t0)

if m.num_solutions:

    for (o , d) in Routes:
        if x[o][d].x >=0.99:
            print(x[o][d], x[o][d].x, f[o][d], f[o][d].x)

    print(m.objective_value)
    print(demand)


#get the routes
res = []
for (o , d) in Routes:
        if x[o][d].x >=0.99:
            scs = x[o][d].name.split('_')[1:]
            res.append(scs)

# get available xi 
res_num = []
for (o , d) in Routes:
        if x[o][d].x >=0.99:
            res_num.append(x[o][d].x)        
            
#get flows...  
Flows_of_Routes = []
for (o , d) in Routes:
    if x[o][d].x >=0.99:
        Flows_of_Routes.append([f[o][d].name.split('_')[1], f[o][d].name.split('_')[2], f[o][d].x])
                
for i in range(len(Flows_of_Routes)):
    Flows_of_Routes[i].append(res_num[i])


point = 'loc1'
available = sum(res_num) 
route = []
pd = []
pop_fl_list = []
start_load = 0
flows_list = [flows[:] for flows in Flows_of_Routes]

G = defaultdict(list)
for (s,t) in res:
    G[s].append(t)
    # G[t].append(s)

G1 =   nx.DiGraph()
for (i, j) in res:
    G1.add_edge(i, j, weight =f[i][j].x )

edge_labels=dict([((u,v,),d['weight']) for u,v,d in G1.edges(data=True)])
pos=nx.spring_layout(G1)
nx.draw(G1,pos, with_labels = True)


plt.show()

