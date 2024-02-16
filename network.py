import numpy as np
import nengo

class StimulusNode:
    def __init__(self):
        self.stimulus_time = 0
        self.location = 0
        self.direction = 0
        
        self.in_center = False

    def step(self, t):
        if not self.in_center: #if not in center state: get random values for location & direction
            if t > self.stimulus_time:
                location, direction = self.location, self.direction
                self.location, self.direction = np.random.choice([-1, 1], size=2)

                self.stimulus_time += 2.0
                self.in_center = True
            return self.location, self.direction
        else:
            if t > self.stimulus_time: 
                self.in_center = False 
                self.stimulus_time += 2.0

            #reset location and direction; no stimuli are presented
            return 0, 0

class Memory:
    def __init__(self):
        self.stimulus_time = 0
        self.congruency_prev = 0
        self.first_values = True
        self.in_center = False
        self.response_prev = 0

    def update_goal(self, t, x):
        if not self.in_center: #if not in center state: get random values for location & direction
            if t > self.stimulus_time:
                
                if (self.first_values): #congurency is 0 for the first trial
                    congruency = 0
                    self.first_values = False
                    
                else:
                    location, direction = x
                    if (round(direction) == round(location)): #check for congruency
                        congruency = 1
                    else:
                        congruency = 0

                    
                self.congruency_prev = congruency #remember congruency for next trial
                                
                
                self.stimulus_time += 2.0
                self.in_center = True
                
            return self.congruency_prev
        else:
            if t > self.stimulus_time: 
                self.in_center = False 
                self.stimulus_time += 2.0

            return 0
        
    def update_response(self, t, x):
        if not self.in_center: #if not in center state: get random values for location & direction
            if t > self.stimulus_time:
                if (self.first_values): #congurency is 0 for the first trial
                    self.response_prev = 0
                    self.first_values = False
                else:
                    response = x
                    self.response_prev = response
                                
                self.stimulus_time += 2.0
                self.in_center = True
                
            return self.response_prev
        else:
            if t > self.stimulus_time: 
                self.in_center = False 
                self.stimulus_time += 2.0

            return 0
class Regulate:
    def __init__(self):
        self.congruency = 0
        self.stimulus_time = 0
        self.in_center = False
    def update_sign(self, t, x):
        if not self.in_center: #if not in center state: get random values for location & direction
            if t > self.stimulus_time:
                congruency_prev, direction = x
                if direction > 0:
                    congruency = np.abs(congruency_prev)
                elif direction < 0:
                    congruency = -np.abs(congruency_prev)
                else: 
                    congruency = congruency_prev
                self.congruency = congruency
                return self.congruency
        else:
            if t > self.stimulus_time: 
                self.in_center = False 
                self.stimulus_time += 2.0

            return 0
model = nengo.Network()
with model:
    
    input_loc = nengo.Ensemble(n_neurons=100, dimensions=1)
    input_dir = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    goal_layer = nengo.Ensemble(n_neurons=100, dimensions=2)
    response_layer = nengo.Ensemble(n_neurons=200, dimensions=4, neuron_type=nengo.LIF())
    decision_ens = nengo.Ensemble(n_neurons=200, dimensions=1, neuron_type=nengo.Sigmoid())
    arm_ens = nengo.Ensemble(n_neurons=200, dimensions=2)
    
    with nengo.Network("Memory"):
        mem_node = nengo.Node(Memory().update_goal, size_in=2, size_out=1, label="congruencyN-1")
        response_node = nengo.Node(Memory().update_response, size_in=1, size_out=1, label="responseN-1")
    with nengo.Network("Signal_regulation"):
        regulation_node = nengo.Node(Regulate().update_sign, size_in = 2, size_out=1, label="regulate sign")
        
    with nengo.Network("Stimuli"):
        stim_node = nengo.Node(StimulusNode().step, label="stimulus")
        
        def map_position(x):
            return x             
    
        # Stimuli -> Input layers
        nengo.Connection(stim_node[0], input_loc)
        nengo.Connection(stim_node[1], input_dir)
        
        # Input layers -> Goal layer        
        nengo.Connection(input_loc, goal_layer[0], synapse=0.1)
        nengo.Connection(input_dir, goal_layer[1], synapse=0.1)
        
        #Input layers -> Response layer
        nengo.Connection(input_loc, response_layer[0], synapse=0.01)
        nengo.Connection(input_dir, response_layer[1], synapse=0.2)
        
        #Goal layer feeds Memory
        nengo.Connection(goal_layer, mem_node, synapse=0.1)
        
        # Response Layer calls Memory + Regulation node for congruencyN-1
        nengo.Connection(mem_node, regulation_node[0], synapse=None)
        nengo.Connection(response_layer[1], regulation_node[1], synapse=None)
        nengo.Connection(regulation_node, response_layer[2], synapse=0.01)
        
        #nengo.Connection(mem_node, response_layer[2], synapse=0.01)
        #nengo.Connection(response_layer, response_layer, function=product_func, synapse=None)
        
        #Response Layer calls Memory for responseN-1
        nengo.Connection(response_node, response_layer[3], synapse=0.01)
        
        conn = nengo.Connection(response_layer, decision_ens, 
                                transform=[[0.3, 0.9, 0.15, 0.01]],
                                learning_rule_type=nengo.RLS(learning_rate=1e-6))
        
        # Connect goal_layer to the learning rule        
        nengo.Connection(goal_layer[1], conn.learning_rule, transform=-1)

        #Decision -> Memory
        nengo.Connection(decision_ens, response_node, synapse=0.1)
        
        #Decision -> Arm
        nengo.Connection(decision_ens, arm_ens[0], function=map_position, synapse=0.1)
        nengo.Connection(decision_ens, arm_ens[1], function=lambda x: 1 if np.round(x) != 0 else 0, synapse=0.1)
        
    input_stimulus = nengo.Probe(stim_node, synapse=0.1)
    signals_probe = nengo.Probe(response_layer, synapse=0.1)
    continuous_decision = nengo.Probe(decision_ens, synapse=0.1)

with nengo.Simulator(model) as sim:
    for trial in range(num_trials):
        sim.run(trial_duration+rest_duration)
        #stimulus data for the current trial
        trial_stimulus_data = sim.data[input_stimulus]
        
        #last value of the stimulus
        last_stimulus_values = trial_stimulus_data[-1]
        
        # Calculate congruency
        if (round(last_stimulus_values[0])!=0):
            congruency = 1 if round(last_stimulus_values[0]) == round(last_stimulus_values[1]) else 0
        
        # add the congruency value to dict
        if congruency==1:
            congruency_values["congruent"].append(trial)
        else:
            congruency_values["incongruent"].append(trial)
        
        