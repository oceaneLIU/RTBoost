import json
import copy
import time
import random
import numpy as np
import Levenshtein
import pandas as pd
from tqdm import tqdm
import networkx as nx
from functools import partial
import tiktoken
import logging
import time
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import re
from scipy.optimize import linear_sum_assignment
from rdfs.dependency_graph.models.language import Language
from rdfs.dependency_graph import construct_dependency_graph
from rdfs.dependency_graph.models.repository import Repository
from rdfs.dependency_graph.graph_generator import GraphGeneratorType
from rdfs.dependency_graph import construct_dependency_graph, DependencyGraph
from rdfs.dependency_graph.models.graph_data import Node, NodeType, EdgeRelation
from tree_sitter import Parser, Language as TS_Language, Node as TS_Node, Tree
from rdfs.dependency_graph.graph_generator.tree_sitter_generator.load_lib import get_builtin_lib_path


def get_key_words(lang: Language):
    if lang == Language.Java:
        f_name = "./bleu/CodeBLEU/keywords/java.txt"
    elif lang == Language.CSharp:
        f_name = "./bleu/CodeBLEU/keywords/c_sharp.txt"
    elif lang == Language.ArkTS:
        f_name = "./bleu/CodeBLEU/keywords/arkts.txt"
    else:
        f_name = ""
    with open(f_name, 'r') as f:
        res = f.readlines()
    return [r.strip() for r in res]


def get_identifiers(lang, source_code):
    parser = Parser()
    lib_path = get_builtin_lib_path()
    ts_language = TS_Language(str(lib_path.absolute()), lang)
    parser.set_language(ts_language)
    tree = parser.parse(source_code.encode())
    keyword_list = get_key_words(lang)

    def dfs_identifier(node):
        identifiers = []
        if node.type == "identifier":
            node_code = source_code[node.start_byte:node.end_byte].decode("utf8")
            if node_code not in keyword_list:
                identifiers.append(node_code.lower())
        for child in node.children:
            identifiers.extend(get_identifiers(child, source_code))
        return identifiers

    return dfs_identifier(tree.root_node)


def get_similarity(text1, text2, lang1, lang2):

    set1 = set(get_identifiers(lang1, text1))
    set2 = set(get_identifiers(lang2, text2))

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return float(intersection) / union


class GraphAligner:

    def __init__(self, graph1: nx.MultiDiGraph, graph2: nx.MultiDiGraph,
                 lang1, lang2, alpha=0.8):
        self.graph1 = graph1
        self.graph2 = graph2
        self.keywords1 = get_key_words(lang1)
        self.keywords2 = get_key_words(lang2)
        self.all_node_types = set()
        self.graph_node1 = dict()
        self.graph_node2 = dict()
        self.alpha = alpha
        self.nodeid1 = dict()
        self.nodeid2 = dict()
        self.lang1 = lang1
        self.lang2 = lang2
        nodes1 = list(self.graph1.nodes)
        nodes1.sort(key=lambda x: x.__str__())
        nodes2 = list(self.graph2.nodes)
        nodes2.sort(key=lambda x: x.__str__())
        for i, v in enumerate(nodes1):
            self.graph_node1[i] = v
            self.nodeid1[v] = i
            self.all_node_types.add(v.type)
        for i, v in enumerate(nodes2):
            self.graph_node2[i] = v
            self.nodeid2[v] = i
            self.all_node_types.add(v.type)
        self.node_type1 = dict()
        self.node_type2 = dict()
        for node_type in self.all_node_types:
            self.node_type1[node_type] = []
            self.node_type2[node_type] = []
        for i in range(0, len(self.graph_node1)):
            v = self.graph_node1[i]
            t = v.type
            if t == NodeType.CONSTRUCTOR:
                self.node_type1[NodeType.METHOD].append(i)
            elif t == NodeType.REPO:
                self.node_type1[NodeType.DIR].append(i)
            else:
                self.node_type1[t].append(i)
        for i in range(0, len(self.graph_node2)):
            v = self.graph_node2[i]
            t = v.type
            if t == NodeType.CONSTRUCTOR:
                self.node_type2[NodeType.METHOD].append(i)
            elif t == NodeType.REPO:
                self.node_type2[NodeType.DIR].append(i)
            else:
                self.node_type2[t].append(i)

    def get_single_type_initial_alignment(self, t):
        if len(self.node_type1[t]) == 0 or len(self.node_type2[t]) == 0:
            return [], 0
        res = []
        sim_matrix = np.zeros((len(self.node_type1[t]), len(self.node_type2[t])))
        for i in range(0, len(self.node_type1[t])):
            for j in range(0, len(self.node_type2[t])):
                v = self.node_type1[t][i]
                u = self.node_type2[t][j]
                sim_matrix[i][j] = get_similarity(self.graph_node1[v].name, self.graph_node2[u].name,
                                                  self.lang1, self.lang2)
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        sum_sim = sim_matrix[row_ind, col_ind].sum()
        for i in range(0, len(row_ind)):
            v = self.node_type1[t][row_ind[i]]
            u = self.node_type2[t][col_ind[i]]
            res.append((v, u))
        return res, sum_sim

    def get_node_scmn(self):
        node_com1 = dict()
        node_com2 = dict()
        all_edge_relation = set()
        for v in self.graph1.nodes:
            idv = self.nodeid1[v]
            node_com1[idv] = set()
        for u in self.graph2.nodes:
            idu = self.nodeid2[u]
            node_com2[idu] = set()
        for v in self.graph1.nodes:
            idv = self.nodeid1[v]
            for u, _, t in self.graph1.in_edges(v, data="relation"):
                feature_str = f"-;{t.relation}@{u.name}"
                node_com1[idv].add(feature_str)
                all_edge_relation.add(t.relation)
            for _, u, t in self.graph1.out_edges(v, data="relation"):
                feature_str = f"+;{t.relation}@{u.name}"
                node_com1[idv].add(feature_str)
                all_edge_relation.add(t.relation)
        for v in self.graph2.nodes:
            idv = self.nodeid2[v]
            for u, _, t in self.graph2.in_edges(v, data="relation"):
                feature_str = f"-;{t.relation}@{u.name}"
                node_com2[idv].add(feature_str)
                all_edge_relation.add(t.relation)
            for _, u, t in self.graph2.out_edges(v, data="relation"):
                feature_str = f"+;{t.relation}@{u.name}"
                node_com2[idv].add(feature_str)
                all_edge_relation.add(t.relation)
        return node_com1, node_com2, all_edge_relation

    def calculate_commonality(self, node_com1: set, node_com2: set, all_edge_relation: set):
        com = 0
        cnt = 0
        if len(node_com1) != 0 and len(node_com2) != 0:
            for k in all_edge_relation:
                f1_names = []
                f2_names = []
                for s in node_com1:
                    if s.startswith(f"-;{k}@"):
                        f1_names.append(s.removeprefix(f"-;{k}@"))
                for s in node_com2:
                    if s.startswith(f"-;{k}@"):
                        f2_names.append(s.removeprefix(f"-;{k}@"))
                f1_names.sort()
                f2_names.sort()
                f1_names_str = " ".join(f1_names)
                f2_names_str = " ".join(f2_names)
                com += get_similarity(f1_names_str, f2_names_str, self.lang1, self.lang2)
                if len(f1_names) != 0 or len(f2_names) != 0:
                    cnt += 1
                f1_names = []
                f2_names = []
                for s in node_com1:
                    if s.startswith(f"+;{k}@"):
                        f1_names.append(s.removeprefix(f"+;{k}@"))
                for s in node_com2:
                    if s.startswith(f"+;{k}@"):
                        f2_names.append(s.removeprefix(f"+;{k}@"))
                f1_names.sort()
                f2_names.sort()
                f1_names_str = " ".join(f1_names)
                f2_names_str = " ".join(f2_names)
                com += get_similarity(f1_names_str, f2_names_str, self.lang1, self.lang2)
                if len(f1_names) != 0 or len(f2_names) != 0:
                    cnt += 1
            return com / cnt
        else:
            return 0

    def get_membership_successors(self, graph, node, node_type):
        children = []
        queue = [target for source, target, data in graph.out_edges(node, data=True) if
                 data['relation'].relation == EdgeRelation.HasMember and target.type != node.type]
        visited = set()
        while queue:
            v = queue.pop(0)
            visited.add(v)
            if node_type == NodeType.METHOD:
                if v.type in [NodeType.METHOD, NodeType.CONSTRUCTOR]:
                    children.append(v)
            else:
                if v.type == node_type:
                    children.append(v)
            for _, target, data in graph.out_edges(v, data=True):
                if data['relation'].relation == EdgeRelation.HasMember and target.type != node.type:
                    if target not in visited:
                        queue.append(target)
        return children

    def bfs_alignment(self, node_com1, node_com2, all_edge_relation, level, node_pair):
        v = node_pair[0]
        u = node_pair[1]
        children_v = self.get_membership_successors(self.graph1, v, level)
        children_u = self.get_membership_successors(self.graph2, u, level)
        alignment = []
        sum_sim = 0

        sim_matrix = np.zeros((len(children_v), len(children_u)))
        for i in range(0, len(children_v)):
            for j in range(0, len(children_u)):
                v = children_v[i]
                u = children_u[j]
                if level == NodeType.METHOD:
                    sig1 = v.content.split('{', maxsplit=1)[0].strip()
                    sig2 = u.content.split('{', maxsplit=1)[0].strip()
                    sig1 = v.name + " " + sig1.split('(')[-1].rstrip(')')
                    sig2 = u.name + " " + sig2.split('(')[-1].rstrip(')')
                    sim_matrix[i][j] = self.alpha * get_similarity(sig1, sig2, self.lang1, self.lang2)
                else:
                    sim_matrix[i][j] = self.alpha * get_similarity(v.name, u.name, self.lang1, self.lang2)
                sim_matrix[i][j] += (1 - self.alpha) * self.calculate_commonality(node_com1[self.nodeid1[v]],
                                                                                  node_com2[self.nodeid2[u]],
                                                                                  all_edge_relation)
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        sum_sim += sim_matrix[row_ind, col_ind].sum()
        for i in range(0, len(row_ind)):
            v = children_v[row_ind[i]]
            u = children_u[col_ind[i]]
            alignment.append((v, u))

        return alignment

    def get_layerwise_alignment(self):
        node_com1, node_com2, all_edge_relation = self.get_node_scmn()
        alignment = []

        curr_t = NodeType.DIR

        sum_sim = 0
        if len(self.node_type1[curr_t]) != 0 or len(self.node_type2[curr_t]) != 0:
            sim_matrix = np.zeros((len(self.node_type1[curr_t]), len(self.node_type2[curr_t])))
            for i in range(0, len(self.node_type1[curr_t])):
                for j in range(0, len(self.node_type2[curr_t])):
                    v = self.node_type1[curr_t][i]
                    u = self.node_type2[curr_t][j]
                    sim_matrix[i][j] = self.alpha * get_similarity(self.graph_node1[v].name, self.graph_node2[u].name,
                                                                   self.lang1, self.lang2)
                    sim_matrix[i][j] += (1 - self.alpha) * self.calculate_commonality(node_com1[v], node_com2[u],
                                                                                      all_edge_relation)
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)
            sum_sim += sim_matrix[row_ind, col_ind].sum()
            for i in range(0, len(row_ind)):
                v = self.node_type1[curr_t][row_ind[i]]
                u = self.node_type2[curr_t][col_ind[i]]
                alignment.append((self.graph_node1[v], self.graph_node2[u]))

        max_processes = os.cpu_count()
        worker_partial = partial(self.bfs_alignment, node_com1, node_com2, all_edge_relation, NodeType.FILE)
        with ThreadPoolExecutor(3 * max_processes) as executor:
            res = list(tqdm(executor.map(worker_partial, alignment), total=len(alignment),
                            desc="Align file level"))
        file_alignments = []
        for a in res:
            file_alignments += copy.deepcopy(a)

        worker_partial = partial(self.bfs_alignment, node_com1, node_com2, all_edge_relation, NodeType.CLASS)
        with ThreadPoolExecutor(3 * max_processes) as executor:
            res = list(tqdm(executor.map(worker_partial, file_alignments), total=len(file_alignments),
                            desc="Align class level"))
        class_alignments = []
        for a in res:
            class_alignments += copy.deepcopy(a)

        worker_partial = partial(self.bfs_alignment, node_com1, node_com2, all_edge_relation, NodeType.METHOD)
        with ThreadPoolExecutor(3 * max_processes) as executor:
            res = list(tqdm(executor.map(worker_partial, class_alignments), total=len(class_alignments),
                            desc="Align method level"))
        method_alignments = []
        for a in res:
            method_alignments += copy.deepcopy(a)

        worker_partial = partial(self.bfs_alignment, node_com1, node_com2, all_edge_relation, NodeType.FIELD)
        with ThreadPoolExecutor(3 * max_processes) as executor:
            res = list(tqdm(executor.map(worker_partial, class_alignments), total=len(class_alignments),
                            desc="Align field level"))
        field_alignments = []
        for a in res:
            field_alignments += copy.deepcopy(a)

        return alignment + file_alignments + class_alignments + method_alignments + field_alignments