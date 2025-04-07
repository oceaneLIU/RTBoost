import networkx as nx
from graph_aligner import GraphAligner, get_similarity
from rdfs.dependency_graph.models.language import Language
from rdfs.dependency_graph import construct_dependency_graph
from rdfs.dependency_graph.models.repository import Repository
from rdfs.dependency_graph.graph_generator import GraphGeneratorType
from rdfs.dependency_graph.models.graph_data import Node, NodeType, EdgeRelation


class Retriever:

    def __init__(self, repo1_path, repo1_language, repo2_path, repo2_language, alpha):
        self.repo1 = Repository(repo1_path, repo1_language)
        self.repo1_language = repo1_language
        self.repo2 = Repository(repo2_path, repo2_language)
        self.repo2_language = repo2_language
        dependency_graph_generator = GraphGeneratorType.TREE_SITTER
        self.graph1 = construct_dependency_graph(self.repo1, dependency_graph_generator)
        self.graph2 = construct_dependency_graph(self.repo2, dependency_graph_generator)
        graph_aligner = GraphAligner(self.graph1.graph, self.graph2.graph, repo1_language, repo2_language, alpha=alpha)
        self.alignment = graph_aligner.get_layerwise_alignment()
        self.language_name = {
            Language.Java: "Java",
            Language.CSharp: "C#",
            Language.ArkTS: "ArkTS",
        }

    def find_method_node(self, method_path: str, method_start_row: int, method_end_row: int, language: Language):
        if language == self.repo1_language:
            nodes = self.graph1.get_nodes(node_filter=lambda node: any([node.type == NodeType.METHOD,
                                                                        node.type == NodeType.CONSTRUCTOR]) and all([
                                                                        str(node.location.file_path) == method_path,
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
                                                                        node.type == NodeType.CONSTRUCTOR]) and all([
                                                                        str(node.location.file_path) == method_path,
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

    def build_dependency_prompt(self, node, graph: nx.MultiDiGraph, alignment_dict: dict):
        dependence_prompt = ""
        dependence_node_list = list(graph.predecessors(node))
        existing_info = set()
        for i, dependence_node in enumerate(dependence_node_list):
            if dependence_node not in alignment_dict.keys():
                continue
            dn = alignment_dict[dependence_node]
            if graph.get_edge_data(dependence_node, node)[0]['relation'].relation.value == EdgeRelation.HasMember:
                continue
            if dn.type in [NodeType.VIRTUAL_API, NodeType.VIRTUAl_PACKAGE, NodeType.CLASS]:
                continue
            if dependence_node == node:
                continue
            if dn.content.strip() not in existing_info:
                existing_info.add(dn.content.strip())
                dependence_prompt += f"\n{dn.content.strip()}\n"
        if dependence_prompt:
            dependence_prompt = f"\nDependencies that used during translation:\n" + f"```{dependence_prompt}```"

        return dependence_prompt

    def build_similar_snippet_prompt(self, node, source_language, target_language):
        content = node.content
        similar_snippet_prompt = "An example of translation pattern:\n"
        max_sim = 0
        max_str_key = ""
        max_str_val = ""
        for a in self.alignment:
            if source_language == self.repo1_language:
                graph_ind = 0
                lang = self.repo1_language
            else:
                graph_ind = 1
                lang = self.repo2_language
            if a[graph_ind].type not in [NodeType.METHOD, NodeType.CONSTRUCTOR]:
                continue
            sim = get_similarity(content, a[graph_ind].content, lang, lang)
            if sim >= max_sim:
                max_sim = sim
                max_str_key = a[graph_ind].content
                max_str_val = a[1-graph_ind].content
        similar_snippet_prompt += f"```{source_language.lower()}\n{max_str_key.rstrip()}\n```\n"
        similar_snippet_prompt += f"```{target_language.lower()}\n{max_str_val.rstrip()}\n```"
        return similar_snippet_prompt

    def build_prompt(self, node: Node, source_language: str, target_language: str):

        translate_prompt = f"Based on the given possible usages and translation pattern, translate the following {self.language_name[source_language]} code to {self.language_name[target_language]} code and explain. Print the {self.language_name[target_language]} code in markdown format(```\\n ```).\n"
        code_to_be_translated = f"Code to be translated:\n```{self.language_name[source_language].lower()}\n{node.content}\n```"

        if source_language == self.repo1_language:
            alignment_dict = dict()
            for v, u in self.alignment:
                alignment_dict[v] = u
            dependence_prompt = self.build_dependency_prompt(node, self.graph1.graph, alignment_dict)
        elif source_language == self.repo2_language:
            alignment_dict = dict()
            for v, u in self.alignment:
                alignment_dict[u] = v
            dependence_prompt = self.build_dependency_prompt(node, self.graph2.graph, alignment_dict)
        else:
            raise NotImplementedError
        similar_snippet_prompt = self.build_similar_snippet_prompt(node, source_language, target_language)
        return translate_prompt.strip() + "\n" + code_to_be_translated + "\n" + dependence_prompt + '\n\n' + similar_snippet_prompt

    def retrival_prompt(self, method_path: str, method_start_row: int, method_end_row: int, source_language: Language, target_language: Language):
        node = self.find_method_node(method_path, method_start_row, method_end_row, source_language)
        prompt = self.build_prompt(node, source_language, target_language)
        return prompt
