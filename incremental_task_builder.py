import networkx as nx
from rdfs.dependency_graph.models.language import Language
from rdfs.dependency_graph.models.repository import Repository
from rdfs.dependency_graph.graph_generator import GraphGeneratorType
from rdfs.dependency_graph.models.graph_data import NodeType
from rdfs.dependency_graph import construct_dependency_graph
import copy


class IncrementalTaskBuilder:

    def __init__(self, repo1_name, repo1_path, repo1_language,
                 repo2_name, repo2_path, repo2_language,
                 repo_trans_task: dict, source_language, target_language):

        self.repo1 = Repository(repo1_path, repo1_language)
        self.repo2 = Repository(repo2_path, repo2_language)
        self.repo1_language = repo1_language
        self.repo2_language = repo2_language
        dependency_graph_generator = GraphGeneratorType.TREE_SITTER
        self.graph1 = construct_dependency_graph(self.repo1, dependency_graph_generator)
        self.graph2 = construct_dependency_graph(self.repo2, dependency_graph_generator)

        self.language_name = {
            Language.Java: "Java",
            Language.CSharp: "C#",
            Language.ArkTS: "ArkTS",
            Language.C: "C",
            Language.Rust: "Rust",
            Language.Python: "Python"
        }
        self.source_language = source_language
        self.target_language = target_language
        self.tasks = repo_trans_task
        self.task_nodes = dict()
        self.task_sorted_list = self.get_topological_sorted_tasks()

    def find_method_node(self, method_path: str, method_start_row: int, method_end_row: int, language: Language):
        if language == self.repo1_language:
            nodes = self.graph1.get_nodes(node_filter=lambda node: any([node.type == NodeType.METHOD,
                                                                        node.type == NodeType.FUNCTION,
                                                                        node.type == NodeType.CONSTRUCTOR]) and all([
                method_path in str(node.location.file_path),
                node.location.start_line <= method_start_row,
                node.location.end_line == method_end_row
            ]))
            if len(nodes) != 1:
                raise ValueError(f"Expected to find one method node, but found {len(nodes)}, method_path: {method_path}"
                                 f", method_start_row: {method_start_row}, method_end_row: {method_end_row}")
            else:
                return nodes[0]
        elif language == self.repo2_language:
            nodes = self.graph2.get_nodes(node_filter=lambda node: any([node.type == NodeType.METHOD,
                                                                        node.type == NodeType.FUNCTION,
                                                                        node.type == NodeType.CONSTRUCTOR]) and all([
                method_path in str(node.location.file_path),
                node.location.start_line <= method_start_row,
                node.location.end_line == method_end_row
            ]))
            if len(nodes) != 1:
                raise ValueError(f"Expected to find one method node, but found {len(nodes)}, method_path: {method_path}"
                                 f", method_start_row: {method_start_row}, method_end_row: {method_end_row}")
            else:
                return nodes[0]
        else:
            raise NotImplementedError

    def get_topological_sorted_tasks(self):
        """return of list of task ind"""
        for task_ind in self.tasks.keys():

            if self.source_language == Language.Java:
                if self.target_language == Language.TypeScript:
                    self.task_nodes[task_ind] = self.find_method_node(
                        self.tasks[task_ind][5].removeprefix("/Users/weiliu/Desktop/android2harmony_dataset/"),
                        self.tasks[task_ind][6],
                        self.tasks[task_ind][7], Language.TypeScript)
                    graph = self.graph2.graph
                elif self.target_language == Language.CSharp:
                    self.task_nodes[task_ind] = self.find_method_node("../java2csharp/" + self.tasks[task_ind][7],
                                                                      self.tasks[task_ind][8],
                                                                      self.tasks[task_ind][9], Language.CSharp)
                    graph = self.graph2.graph
            elif self.source_language == Language.CSharp:
                self.task_nodes[task_ind] = self.find_method_node("../java2csharp/" + self.tasks[task_ind][1],
                                                                  self.tasks[task_ind][2],
                                                                  self.tasks[task_ind][3], Language.Java)
                graph = self.graph1.graph
            elif self.source_language == Language.TypeScript:
                self.task_nodes[task_ind] = self.find_method_node(
                    self.tasks[task_ind][1].removeprefix("/Users/weiliu/Desktop/android2harmony_dataset/"),
                    self.tasks[task_ind][2],
                    self.tasks[task_ind][3], Language.Java)
                graph = self.graph1.graph

        task_graph = nx.DiGraph()
        for v in self.tasks.keys():
            task_graph.add_node(v)
            for u in self.tasks.keys():
                if u == v or self.task_nodes[v].name == self.task_nodes[u].name:
                    continue
                has_edge = False
                if nx.has_path(graph, self.task_nodes[v], self.task_nodes[u]):
                    has_edge = True
                if has_edge:
                    task_graph.add_edge(v, u)

        dag = task_graph.copy()

        while not nx.is_directed_acyclic_graph(dag):
            try:
                cycle = list(nx.find_cycle(dag, orientation="original"))
                print(f"Detected cycle: {cycle}")

                in_degrees = {node: dag.in_degree(node) for node, _, _ in cycle}

                node_to_remove = max(in_degrees, key=in_degrees.get)
                print(f"Removing all incoming edges to node: {node_to_remove}")

                for u in list(dag.predecessors(node_to_remove)):
                    dag.remove_edge(u, node_to_remove)
            except nx.NetworkXNoCycle:
                break

        return list(nx.topological_sort(dag))

    def construct_incremental_translation_task(self, order_in_sorted_tasks: int, language):
        if language == self.repo1_language:
            new_graph = copy.deepcopy(self.graph1)
        elif language == self.repo2_language:
            new_graph = copy.deepcopy(self.graph2)
        for i in range(order_in_sorted_tasks, len(self.task_sorted_list)):
            task_id = self.task_sorted_list[i]
            task_node = self.task_nodes[task_id]
            if task_node in new_graph.graph.nodes:
                dependency_nodes = set(nx.descendants(new_graph.graph, task_node))
                new_graph.graph.remove_nodes_from(dependency_nodes)
                new_graph.graph.remove_node(task_node)
        if language == self.repo1_language:
            return new_graph, self.graph2
        elif language == self.repo2_language:
            return self.graph1, new_graph