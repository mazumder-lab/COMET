# Contains the main MTL MoE model to train, evaluate, and test
import json
from munch import munchify
import tensorflow as tf

from model.experts import ExpertsMapper
from model.gates import GatesMapper
from model.task_specific_heads import HeadsMapper
from model.shared_bottoms import BottomsMapper
from model.permutations import PermutationsMapper

class MoE(tf.keras.Model):

    def __init__(
        self,
        config
    ):
        super(MoE, self).__init__()
                
        # dictionary of the form {expert1_string_identifier: expert1_config} (one for each expert)
        self.experts_config = config["experts"]
        # dictionary of the form {head1_string_identifier: head1_config} (one for each task)    
        self.heads_config = config["heads"]
        # dictionary of the form {gate1_string_identifier: gate1_config} (one for each task, same order as heads)
        if 'k' not in config["gates"][0]["params"]:
            for i in range(len(config["gates"])):
                config["gates"][i]["params"]["k"] = config["gates"][i]["params"]["num_nonzeros"]
        self.gates_config = config["gates"]
        self.k = self.gates_config[0]["params"]["k"]
        # dictionary of the form {perm1_string_identifier: perm1_config} (shared across tasks)
        self.permutations_config = config["permutations"]
        self.permutations_config[0]["params"]['k'] = self.k
        self.permutations_config[0]["params"]['nb_experts'] = self.experts_config[0]['instances']
        # dictionary of the form {bottom_string_identifier: bottom_config} (just one)
        self.bottom_config = config["bottom"]
        self.taskset_id = config["taskset_id"]

        self.bottom = BottomsMapper[self.bottom_config["module"]](self.bottom_config["params"])
        self.heads = self._create_module_list(self.heads_config, HeadsMapper)   
        self.experts = self._create_module_list(self.experts_config, ExpertsMapper)
        self.gates = self._create_module_list(self.gates_config, GatesMapper)
        self.permutations = self._create_module_list(self.permutations_config, PermutationsMapper)
        print("========self.permutations:", self.permutations)
        assert(len(self.gates) == len(self.heads))
        
        self.nb_experts = (int)(self.experts_config[0]['instances'])
        self.nb_gates = (int)(self.gates_config[0]['instances'])
        if "learn_k_permutations" in self.permutations_config[0]:
            if self.permutations_config[0]["learn_k_permutations"]:
                self.no_of_permutations_per_task = (int)(self.permutations_config[0]["params"]['k'])
            else:
                self.no_of_permutations_per_task = 1
        else:
            self.no_of_permutations_per_task = 1
        if self.permutations_config[0]["instances"]==1:
            self.no_of_permutations_all_tasks = self.no_of_permutations_per_task
            self.permutation_per_task = False
        else:
            self.no_of_permutations_all_tasks = self.no_of_permutations_per_task * self.permutations_config[0]["instances"]
            self.permutation_per_task = True
        
        print("==========self.nb_experts:", self.nb_experts)
        print("==========self.nb_gates:", self.nb_gates)
        print("==========self.k:", self.k)
        print("==========self.no_of_permutations_per_task:", self.no_of_permutations_per_task)
        print("==========self.no_of_permutations_all_tasks:", self.no_of_permutations_all_tasks)
        

    def call(
        self,
        x,
        indices=None
    ):
        # h1: (bs, dim_h1)
        h1 = self.bottom(x)
#         tf.print("x:", tf.shape(x), "h1:", tf.shape(h1))
        
        # h2: [(bs, dim_h_exp) for i in range(nb_experts)]
        h2 = [
            expert(h1) for expert in self.experts
        ]
        
        permutations = [perm(x) for perm in self.permutations]
        
        trace_RRT = tf.linalg.trace(
            tf.matmul(
                permutations[0],
                tf.transpose(permutations[0], perm=[0,2,1])
            )
        )
        trace_RTR = tf.linalg.trace(
            tf.matmul(
                tf.transpose(permutations[0], perm=[0,2,1]),
                permutations[0]
            )
        )
        self.add_metric(tf.reduce_mean(trace_RRT), name='trace_RRT')
        self.add_metric(tf.reduce_mean(trace_RTR), name='trace_RTR')
        
        
        # h3: [(bs, dim_h_exp) for i in range(nb_gates)]
        h3 = []
        
        if self.taskset_id in {2,4,11,17,18}:
            if len(h1.shape) > 2:
                h1 = tf.reshape(h1, [h1.shape[0], -1])
            if self.permutation_per_task: 
                for gate, perm in zip(self.gates, permutations):
                    h3_temp = gate((h2, h1, perm), indices=indices)
                    h3.append(h3_temp)                
            else:
                for gate in self.gates:
                    h3_temp = gate((h2, h1, permutations[0]), indices=indices)
                    h3.append(h3_temp)
        else:
            if len(x.shape) > 2:
                x = tf.reshape(x, [x.shape[0], -1])
            if self.permutation_per_task:
                for gate, perm in zip(self.gates, permutations):
                    h3_temp = gate((h2, x, perm), indices=indices)
                    h3.append(h3_temp)
            else:
                for gate in self.gates:
                    h3_temp = gate((h2, x, permutations[0]), indices=indices)
                    h3.append(h3_temp)
        
        # y: [(bs, o_size) for i in range(nb_gates)]
        y = [
            head(h3_g) for h3_g, head in zip(h3, self.heads)
        ]
        #tf.print("\nheads output: ",y)
        
        return y

    
    def _create_module_list(self, config_list, mapper):
        module_list = []
        for m in config_list:
            if mapper == GatesMapper:
                try:
                    m["params"]["nb_experts"] = len(self.experts)
                    print("======len(self.experts):", len(self.experts))
                except:
                    m["params"]["nb_experts"] = self.nb_experts
                    print("======self.nb_experts:", self.nb_experts)
            if m["instances"] > 1:
                for task in range(m["instances"]):
                    if mapper == GatesMapper:
                        m["params"]["task"] = task
                        print(m["params"])
                    elif mapper == PermutationsMapper:
                        m["params"]["task"] = task
                        print(m["params"])
                    module_list.append(mapper[m["module"]](m["params"]))
#                 module_list += [mapper[m["module"]](m["params"]) for _ in range(m["instances"])]
            elif m["instances"] == 1:
                if mapper == GatesMapper:
                    m["params"]["task"] = 0                
                elif mapper == PermutationsMapper:
                    m["params"]["task"] = 0
                module_list.append(mapper[m["module"]](m["params"]))
            else:
                raise ValueError("Incorrect nb of instances specified.")
#         tf.print("=====================module_list:", module_list)
        return module_list