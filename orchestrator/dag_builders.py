# from typing import List, Dict, Any
# from .decomposers import TaskNode

# class BaseDAGBuilder:
#     name: str = "base"
#     def build(self, tasks: List[TaskNode]) -> Dict[str, Any]:
#         raise NotImplementedError

# class ParallelDAGBuilder(BaseDAGBuilder):
#     """Ignores dependencies and returns a single batch to execute simultaneously."""
#     name = "parallel_dag"
    
#     def build(self, tasks: List[TaskNode]) -> Dict[str, Any]:
#         return {"execution_layers": [tasks]} # Everything in layer 1

# class TopologicalDAGBuilder(BaseDAGBuilder):
#     """
#     Builds a true Directed Acyclic Graph (DAG) using Topological Sorting.
#     Groups tasks into sequential execution layers based on dependencies.
#     """
#     name = "topological_dag"
    
#     def build(self, tasks: List[TaskNode]) -> Dict[str, Any]:
#         task_map = {t.id: t for t in tasks}
#         in_degree = {t.id: len(t.depends_on) for t in tasks}
        
#         layers =[]
#         # Find nodes with 0 dependencies
#         ready_nodes =[tid for tid, degree in in_degree.items() if degree == 0]
        
#         while ready_nodes:
#             current_layer = [task_map[tid] for tid in ready_nodes]
#             layers.append(current_layer)
            
#             next_ready =[]
#             for t in tasks:
#                 # If a task depends on a node we just resolved, remove it from its degree
#                 if any(dep in ready_nodes for dep in t.depends_on):
#                     in_degree[t.id] -= sum(1 for dep in t.depends_on if dep in ready_nodes)
#                     if in_degree[t.id] == 0:
#                         next_ready.append(t.id)
                        
#             ready_nodes = next_ready
            
#         return {"execution_layers": layers}

# class DAGRegistry:
#     def __init__(self):
#         self.builders = {
#             "parallel_dag": ParallelDAGBuilder(),
#             "topological_dag": TopologicalDAGBuilder()
#         }
#     def get(self, name: str) -> BaseDAGBuilder:
#         return self.builders.get(name, self.builders["parallel_dag"])

from typing import List, Dict, Any
from collections import defaultdict
from .decomposers import TaskNode

# ---------------------------------------------------------
# BASE INTERFACE
# ---------------------------------------------------------
class BaseDAGBuilder:
    name: str = "base"
    def build(self, tasks: List[TaskNode]) -> Dict[str, Any]:
        raise NotImplementedError

# ---------------------------------------------------------
# 1. PARALLEL DAG (Dumps everything in one layer)
# ---------------------------------------------------------
class ParallelDAGBuilder(BaseDAGBuilder):
    name = "parallel_dag"
    def build(self, tasks: List[TaskNode]) -> Dict[str, Any]:
        return {"execution_layers": [tasks]}

# ---------------------------------------------------------
# 2. TOPOLOGICAL DAG (Classic Kahn's Algorithm)
# ---------------------------------------------------------
class TopologicalDAGBuilder(BaseDAGBuilder):
    name = "topological_dag"
    def build(self, tasks: List[TaskNode]) -> Dict[str, Any]:
        task_map = {t.id: t for t in tasks}
        in_degree = {t.id: len(t.depends_on) for t in tasks}
        
        layers = []
        ready_nodes = [tid for tid, degree in in_degree.items() if degree == 0]
        
        while ready_nodes:
            current_layer = [task_map[tid] for tid in ready_nodes]
            layers.append(current_layer)
            
            next_ready = []
            for t in tasks:
                if any(dep in ready_nodes for dep in t.depends_on):
                    in_degree[t.id] -= sum(1 for dep in t.depends_on if dep in ready_nodes)
                    if in_degree[t.id] == 0:
                        next_ready.append(t.id)
                        
            ready_nodes = next_ready
            
        return {"execution_layers": layers}

# ---------------------------------------------------------
# 3. ADAPTIVE GRAPH-OF-THOUGHTS DAG (The New Engine)
# ---------------------------------------------------------
class GoTDAGBuilder(BaseDAGBuilder):
    """
    Unlike static DAGs, GoT doesn't build pre-computed layers.
    It returns a flag telling the Engine to hand over execution control 
    to the AdaptiveGoTOrchestrator.
    """
    name = "got_dag"
    def build(self, tasks: List[TaskNode]) -> Dict[str, Any]:
        # We just pass the tasks through with a flag
        return {"execution_layers": [tasks], "is_got": True}

# ---------------------------------------------------------
# REGISTRY (The Menu)
# ---------------------------------------------------------
class DAGRegistry:
    def __init__(self):
        self.builders = {
            "parallel_dag": ParallelDAGBuilder(),
            "topological_dag": TopologicalDAGBuilder(),
            "got_dag": GoTDAGBuilder()
        }
        
    def get(self, name: str) -> BaseDAGBuilder:
        return self.builders.get(name, self.builders["topological_dag"])