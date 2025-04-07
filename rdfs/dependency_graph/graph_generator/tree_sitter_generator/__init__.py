import os
from collections import defaultdict, deque
from pathlib import Path
from tree_sitter import Parser, Language as TS_Language, Node as TS_Node, Tree
from tqdm import tqdm
from textwrap import dedent
from rdfs.dependency_graph.dependency_graph import DependencyGraph
from rdfs.dependency_graph.graph_generator import BaseDependencyGraphGenerator
from rdfs.dependency_graph.graph_generator.tree_sitter_generator.finder import (
    ImportFinder, FileFinder, FIND_CLASS_QUERY, FIND_METHOD_QUERY, FIND_FIELD_QUERY, FIND_CLASS_ATTRIBUTE_QUERY,
    FIND_CLASS_USE_QUERY, FIND_METHOD_CALL_QUERY, FIND_FIELD_NAME_QUERY, FIND_IMPORT_ATTRIBUTE_QUERY
)
from rdfs.dependency_graph.graph_generator.tree_sitter_generator.resolve_import import (
    ImportResolver,
)
from rdfs.dependency_graph.graph_generator.tree_sitter_generator.load_lib import (
    get_builtin_lib_path,
)
from rdfs.dependency_graph.models import PathLike
from rdfs.dependency_graph.models.graph_data import (
    Node,
    NodeType,
    Location,
    EdgeRelation,
    Edge,
)
from rdfs.dependency_graph.models.language import Language
from rdfs.dependency_graph.models.repository import Repository
from rdfs.dependency_graph.utils.log import setup_logger
from rdfs.dependency_graph.utils.read_file import read_file_to_string
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

# Initialize logging
logger = setup_logger()


class TreeSitterDependencyGraphGenerator(BaseDependencyGraphGenerator):
    supported_languages: tuple[Language] = (
        Language.Python,
        Language.Java,
        Language.CSharp,
        Language.TypeScript,
        Language.JavaScript,
        Language.Kotlin,
        Language.PHP,
        Language.Ruby,
        Language.C,
        Language.CPP,
        Language.Go,
        Language.Swift,
        Language.Rust,
        Language.Lua,
        Language.Bash,
        Language.R,
    )

    def generate_file(
            self,
            repo: Repository,
            code: str = None,
            file_path: PathLike = None,
    ) -> DependencyGraph:
        raise NotImplementedError("generate_file is not implemented")

    @staticmethod
    def generate_file_node(graph: DependencyGraph, repo: Repository, root_node, suffixes):
        directories = set()
        for file in repo.files:
            path_tuple = file.file_path.relative_to(repo.repo_path).parts
            for index in range(len(path_tuple)):
                directories.add(os.path.join(str(repo.repo_path), '/'.join(path_tuple[:index + 1])))

        def dfs_directory_traversal(dir_path: str, parent_node: Node):

            while os.path.isdir(dir_path):
                sub_file_list = [f for f in os.listdir(dir_path) if not f.startswith('.')]
                available_path_num = 0
                available_path = ""
                for sub_file in sub_file_list:
                    path = os.path.join(dir_path, sub_file)
                    if path in directories:
                        available_path_num += 1
                        avaliable_path = sub_file
                if available_path_num == 1:
                    if os.path.isdir(os.path.join(dir_path, avaliable_path)):
                        dir_path += f"/{avaliable_path}"
                    else:
                        break
                else:
                    break

            name = dir_path.removeprefix(str(parent_node.location.file_path) + "/")
            if any([suffix in name for suffix in suffixes]):
                child_node_type = NodeType.FILE
                name = name.removesuffix("."+name.split(".")[-1])
            else:
                child_node_type = NodeType.DIR

            child_node = Node(type=child_node_type,
                              name=name.strip(),
                              location=Location(file_path=os.path.join(dir_path),
                                                start_line=None, start_column=None,
                                                end_line=None, end_column=None, ),
                              content=name.strip()
                              )
            if not graph.graph.has_node(child_node):
                graph.add_node(child_node)
            graph.add_relational_edge(
                parent_node,
                child_node,
                Edge(
                    relation=EdgeRelation.HasMember,
                    location=None,
                ),
            )

            if os.path.isdir(dir_path):
                files = os.listdir(dir_path)
                for f in files:
                    path = os.path.join(dir_path, f)
                    if path in directories:
                        dfs_directory_traversal(path, child_node)

        for f in os.listdir(str(root_node.location.file_path)):
            full_path = os.path.join(str(root_node.location.file_path), f)
            if full_path in directories:
                dfs_directory_traversal(full_path, root_node)
        return graph

    def generate(self, repo: Repository) -> DependencyGraph:
        D = DependencyGraph(repo.repo_path, repo.language)
        graph = TreeSitterDependencyGraphGenerator.generate_membership_hierarchical(D, repo)
        graph = TreeSitterDependencyGraphGenerator.generate_file_level_dependencies(graph, repo)
        graph = TreeSitterDependencyGraphGenerator.generate_class_level_dependencies(graph, repo)
        graph = TreeSitterDependencyGraphGenerator.generate_method_level_dependencies(graph, repo)
        return graph

    @staticmethod
    def generate_single_file_membership_hierarchical(graph: DependencyGraph, repo: Repository, file_node: Node):
        new_edges = []
        code_context = file_node.get_text()
        if len(code_context.strip()) == 0:
            return

        file_finder = FileFinder(repo.language, code_context)
        tree = file_finder.get_tree()

        class_nodes = file_finder.query_and_captures(FIND_CLASS_QUERY[repo.language], tree.root_node)
        noninternal_class_nodes = []
        for class_node in class_nodes:
            parent = class_node.parent
            if parent:
                parent = parent.parent
                if not (parent and parent.type == 'class_declaration'):
                    noninternal_class_nodes.append(class_node)
            else:
                noninternal_class_nodes.append(class_node)
        new_class_nodes = []
        for class_node in noninternal_class_nodes:
            if class_node.type == 'class_declaration':
                node_type = NodeType.CLASS
            elif class_node.type == 'interface_declaration':
                node_type = NodeType.INTERFACE
            else:
                node_type = NodeType.ENUM

            extends_list = []
            implements_list = []
            use_list = []
            attribute_nodes = file_finder.query_and_captures(FIND_CLASS_ATTRIBUTE_QUERY[repo.language],
                                                             class_node,
                                                             capture_name=['extends', 'implements'])
            for attribute_node, attribute_type in attribute_nodes:
                attribute_location = Location(file_path=file_node.location.file_path,
                                              start_line=attribute_node.start_point[0] + 1,
                                              start_column=attribute_node.start_point[1] + 1,
                                              end_line=attribute_node.end_point[0] + 1,
                                              end_column=attribute_node.end_point[1] + 2)
                attribute_name = attribute_location.get_text()
                if attribute_type == 'extends':
                    extends_list.append(attribute_name.strip())
                elif attribute_type == 'implements':
                    implements_list.append(attribute_name.strip())
            use_nodes = file_finder.query_and_captures(FIND_CLASS_USE_QUERY[repo.language],
                                                       class_node.child_by_field_name('body'))
            for use_node in use_nodes:
                use_location = Location(file_path=file_node.location.file_path,
                                        start_line=use_node.start_point[0] + 1,
                                        start_column=use_node.start_point[1] + 1,
                                        end_line=use_node.end_point[0] + 1,
                                        end_column=use_node.end_point[1] + 2)
                use_name = use_location.get_text()
                use_list.append(use_name.strip())

            name_location = Location(file_path=file_node.location.file_path,
                                     start_line=class_node.child_by_field_name('name').start_point[0] + 1,
                                     start_column=class_node.child_by_field_name('name').start_point[1] + 1,
                                     end_line=class_node.child_by_field_name('name').end_point[0] + 1,
                                     end_column=class_node.child_by_field_name('name').end_point[1] + 2)
            body_start_pos = class_node.child_by_field_name('body').start_point
            if body_start_pos[1] == 0:
                signature_end_line = body_start_pos[0]
                signature_end_column = body_start_pos[1]
            else:
                signature_end_line = body_start_pos[0]
                signature_end_column = body_start_pos[1] - 1
            class_location = Location(file_path=file_node.location.file_path,
                                      start_line=class_node.start_point[0] + 1,
                                      start_column=0,
                                      end_line=signature_end_line + 1,
                                      end_column=signature_end_column + 1)
            extends_list.sort()
            implements_list.sort()
            use_list = list(set(use_list))
            use_list.sort()
            class_name = name_location.get_text().strip().strip('<')
            new_class_node = Node(
                type=node_type,
                name=class_name,
                location=class_location,
                attribute={
                    "extends": extends_list.copy(),
                    "implements": implements_list.copy(),
                    "uses": use_list.copy()
                },
                content=class_location.get_text().rstrip("{").strip(),
            )
            if not graph.graph.has_node(new_class_node):
                graph.add_node(new_class_node)
                new_class_nodes.append(new_class_node)
            new_edges.append((file_node,
                new_class_node,
                Edge(
                    relation=EdgeRelation.HasMember,
                    location=None,
                )))
            field_nodes = file_finder.query_and_captures(FIND_FIELD_QUERY[repo.language], class_node)
            for field_node in field_nodes:
                parent = field_node.parent
                if parent and parent.type == 'declaration_list':
                    parent = parent.parent
                    if parent and parent.type == 'class_declaration':
                        parent_name = parent.child_by_field_name('name').text.decode()
                        if parent_name != class_name:
                            continue
                field_name = list(file_finder.query_and_captures(FIND_FIELD_NAME_QUERY[repo.language], field_node))[
                    0]
                field_name_location = Location(file_path=file_node.location.file_path,
                                               start_line=field_name.start_point[0] + 1,
                                               start_column=field_name.start_point[1] + 1,
                                               end_line=field_name.end_point[0] + 1,
                                               end_column=field_name.end_point[1] + 2, )
                field_location = Location(file_path=file_node.location.file_path,
                                          start_line=field_node.start_point[0] + 1,
                                          start_column=field_node.start_point[1] + 1,
                                          end_line=field_node.end_point[0] + 1,
                                          end_column=field_node.end_point[1] + 2, )
                new_field_node = Node(
                    type=NodeType.FIELD,
                    name=field_name_location.get_text().strip().strip(':'),
                    location=field_location,
                    content=field_location.get_text().strip()
                )
                if not graph.graph.has_node(new_field_node):
                    graph.add_node(new_field_node)
                new_edges.append((
                    new_class_node,
                    new_field_node,
                    Edge(
                        relation=EdgeRelation.HasMember,
                        location=None,
                    )
                ))
            method_nodes = file_finder.query_and_captures(FIND_METHOD_QUERY[repo.language], class_node)
            for method_node in method_nodes:
                parent = method_node.parent
                if parent and parent.type == 'declaration_list':
                    parent = parent.parent
                    if parent and parent.type == 'class_declaration':
                        parent_name = parent.child_by_field_name('name').text.decode()
                        if parent_name != class_name:
                            continue
                if method_node.type in ["method_declaration", "method_definition"]:
                    node_type = NodeType.METHOD
                elif method_node.type == "function_declaration":
                    node_type = NodeType.FUNCTION
                else:
                    node_type = NodeType.CONSTRUCTOR
                method_call_set = file_finder.query_and_captures(FIND_METHOD_CALL_QUERY[repo.language], method_node)
                method_call_list = []
                for method_call in method_call_set:
                    method_call_location = Location(file_path=file_node.location.file_path,
                                                    start_line=method_call.start_point[0] + 1,
                                                    start_column=method_call.start_point[1] + 1,
                                                    end_line=method_call.end_point[0] + 1,
                                                    end_column=method_call.end_point[1] + 2)
                    method_call_name = method_call_location.get_text()
                    method_call_list.append(method_call_name)
                method_call_list.sort()
                name_location = Location(file_path=file_node.location.file_path,
                                         start_line=method_node.child_by_field_name('name').start_point[0] + 1,
                                         start_column=method_node.child_by_field_name('name').start_point[1] + 1,
                                         end_line=method_node.child_by_field_name('name').end_point[0] + 1,
                                         end_column=method_node.child_by_field_name('name').end_point[1] + 1, )
                method_location = Location(file_path=file_node.location.file_path,
                                           start_line=method_node.start_point[0] + 1, start_column=0,
                                           end_line=method_node.end_point[0] + 1,
                                           end_column=method_node.end_point[1] + 2, )
                new_method_node = Node(
                    type=node_type,
                    name=name_location.get_text().strip(),
                    location=method_location,
                    attribute={'method_call_list': method_call_list.copy()},
                    content=dedent(method_location.get_text().strip('\n'))
                )
                if not graph.graph.has_node(new_method_node):
                    graph.add_node(new_method_node)
                new_edges.append((
                    new_class_node,
                    new_method_node,
                    Edge(
                        relation=EdgeRelation.HasMember,
                        location=None,
                    )
                ))

        for i, e_class_node in enumerate(noninternal_class_nodes):
            body_node = e_class_node.child_by_field_name("body")
            class_nodes = file_finder.query_and_captures(FIND_CLASS_QUERY[repo.language], body_node)
            for class_node in class_nodes:
                if class_node.type == 'class_declaration':
                    node_type = NodeType.CLASS
                elif class_node.type == 'interface_declaration':
                    node_type = NodeType.INTERFACE
                else:
                    node_type = NodeType.ENUM

                extends_list = []
                implements_list = []
                use_list = []
                attribute_nodes = file_finder.query_and_captures(FIND_CLASS_ATTRIBUTE_QUERY[repo.language],
                                                                 class_node,
                                                                 capture_name=['extends', 'implements'])
                for attribute_node, attribute_type in attribute_nodes:
                    attribute_location = Location(file_path=file_node.location.file_path,
                                                  start_line=attribute_node.start_point[0] + 1,
                                                  start_column=attribute_node.start_point[1] + 1,
                                                  end_line=attribute_node.end_point[0] + 1,
                                                  end_column=attribute_node.end_point[1] + 2)
                    attribute_name = attribute_location.get_text()
                    if attribute_type == 'extends':
                        extends_list.append(attribute_name.strip())
                    elif attribute_type == 'implements':
                        implements_list.append(attribute_name.strip())
                use_nodes = file_finder.query_and_captures(FIND_CLASS_USE_QUERY[repo.language],
                                                           class_node.child_by_field_name('body'))
                for use_node in use_nodes:
                    use_location = Location(file_path=file_node.location.file_path,
                                            start_line=use_node.start_point[0] + 1,
                                            start_column=use_node.start_point[1] + 1,
                                            end_line=use_node.end_point[0] + 1,
                                            end_column=use_node.end_point[1] + 2)
                    use_name = use_location.get_text()
                    use_list.append(use_name.strip())

                name_location = Location(file_path=file_node.location.file_path,
                                         start_line=class_node.child_by_field_name('name').start_point[0] + 1,
                                         start_column=class_node.child_by_field_name('name').start_point[1] + 1,
                                         end_line=class_node.child_by_field_name('name').end_point[0] + 1,
                                         end_column=class_node.child_by_field_name('name').end_point[1] + 2)
                body_start_pos = class_node.child_by_field_name('body').start_point
                if body_start_pos[1] == 0:
                    signature_end_line = body_start_pos[0]
                    signature_end_column = body_start_pos[1]
                else:
                    signature_end_line = body_start_pos[0]
                    signature_end_column = body_start_pos[1] - 1
                class_location = Location(file_path=file_node.location.file_path,
                                          start_line=class_node.start_point[0] + 1,
                                          start_column=0,  # 为typescript做的优化
                                          end_line=signature_end_line + 1,
                                          end_column=signature_end_column + 1)
                extends_list.sort()
                implements_list.sort()
                use_list = list(set(use_list))
                use_list.sort()
                class_name = name_location.get_text().strip()
                new_class_node = Node(
                    type=node_type,
                    name=class_name,
                    location=class_location,
                    attribute={
                        "extends": extends_list.copy(),
                        "implements": implements_list.copy(),
                        "uses": use_list.copy()
                    },
                    content=class_location.get_text().rstrip("{").strip(),
                )
                if not graph.graph.has_node(new_class_node):
                    graph.add_node(new_class_node)
                new_edges.append((
                    new_class_nodes[i],
                    new_class_node,
                    Edge(
                        relation=EdgeRelation.HasMember,
                        location=None,
                    )
                ))
                field_nodes = file_finder.query_and_captures(FIND_FIELD_QUERY[repo.language], class_node)
                for field_node in field_nodes:
                    field_name = list(file_finder.query_and_captures(FIND_FIELD_NAME_QUERY[repo.language], field_node))[
                        0]
                    field_name_location = Location(file_path=file_node.location.file_path,
                                                   start_line=field_name.start_point[0] + 1,
                                                   start_column=field_name.start_point[1] + 1,
                                                   end_line=field_name.end_point[0] + 1,
                                                   end_column=field_name.end_point[1] + 2, )
                    field_location = Location(file_path=file_node.location.file_path,
                                              start_line=field_node.start_point[0] + 1,
                                              start_column=field_node.start_point[1] + 1,
                                              end_line=field_node.end_point[0] + 1,
                                              end_column=field_node.end_point[1] + 2, )
                    new_field_node = Node(
                        type=NodeType.FIELD,
                        name=field_name_location.get_text().strip().strip(':'),
                        location=field_location,
                        content=field_location.get_text().strip()
                    )
                    if not graph.graph.has_node(new_field_node):
                        graph.add_node(new_field_node)
                    new_edges.append((
                        new_class_node,
                        new_field_node,
                        Edge(
                            relation=EdgeRelation.HasMember,
                            location=None,
                        )
                    ))
                method_nodes = file_finder.query_and_captures(FIND_METHOD_QUERY[repo.language], class_node)
                for method_node in method_nodes:
                    if method_node.type in ["method_declaration", "method_definition"]:
                        node_type = NodeType.METHOD
                    elif method_node.type == "function_declaration":
                        node_type = NodeType.FUNCTION
                    else:
                        node_type = NodeType.CONSTRUCTOR
                    method_call_set = file_finder.query_and_captures(FIND_METHOD_CALL_QUERY[repo.language], method_node)
                    method_call_list = []
                    for method_call in method_call_set:
                        method_call_location = Location(file_path=file_node.location.file_path,
                                                        start_line=method_call.start_point[0] + 1,
                                                        start_column=method_call.start_point[1] + 1,
                                                        end_line=method_call.end_point[0] + 1,
                                                        end_column=method_call.end_point[1] + 2)
                        method_call_name = method_call_location.get_text()
                        method_call_list.append(method_call_name)
                    method_call_list.sort()
                    name_location = Location(file_path=file_node.location.file_path,
                                             start_line=method_node.child_by_field_name('name').start_point[0] + 1,
                                             start_column=method_node.child_by_field_name('name').start_point[1] + 1,
                                             end_line=method_node.child_by_field_name('name').end_point[0] + 1,
                                             end_column=method_node.child_by_field_name('name').end_point[1] + 1, )
                    method_location = Location(file_path=file_node.location.file_path,
                                               start_line=method_node.start_point[0] + 1, start_column=0,
                                               end_line=method_node.end_point[0] + 1,
                                               end_column=method_node.end_point[1] + 2, )
                    new_method_node = Node(
                        type=node_type,
                        name=name_location.get_text().strip(),
                        location=method_location,
                        attribute={'method_call_list': method_call_list.copy()},
                        content=dedent(method_location.get_text().strip('\n'))
                    )
                    if not graph.graph.has_node(new_method_node):
                        graph.add_node(new_method_node)
                    new_edges.append((
                        new_class_node,
                        new_method_node,
                        Edge(
                            relation=EdgeRelation.HasMember,
                            location=None,
                        )
                    ))
        return new_edges

    @staticmethod
    def generate_membership_hierarchical(graph: DependencyGraph, repo: Repository):
        """
        Generate the membership hierarchical graph.等价于文件结构树和文件内部的展开
        :param graph:
        :param repo:
        :return:
        """
        repo_name = repo.repo_path.name
        suffixes = repo.code_file_extensions[repo.language]

        root_node = Node(type=NodeType.REPO,
                         name=repo_name.strip(),
                         location=Location(file_path=repo.repo_path, start_line=None, start_column=None,
                                           end_line=None, end_column=None, ),
                         content=repo_name.strip()
                         )

        graph.add_node(root_node)

        graph = TreeSitterDependencyGraphGenerator.generate_file_node(graph, repo, root_node, suffixes)

        all_node_list = graph.get_nodes(node_filter=lambda node: node.type == NodeType.FILE)

        max_processes = os.cpu_count()
        worker_partial = partial(TreeSitterDependencyGraphGenerator.generate_single_file_membership_hierarchical,
                                 graph, repo)
        with ProcessPoolExecutor(max_processes) as executor:
            all_edges = list(tqdm(executor.map(worker_partial, all_node_list), total=len(all_node_list),
                                  desc="Generating membership hierarchical graph"))
        for edges in all_edges:
            if edges:
                graph.add_relational_edges_from(edges)

        return graph

    @staticmethod
    def is_third_party_import(package_name, api_name, module_map, current_file_path, language):
        match language:
            case Language.Java:
                key_name = f"{package_name}.{api_name}"
                if key_name not in module_map.keys():
                    return True
                importee_file_path = module_map[key_name][0]
                if (not importee_file_path.exists() or not importee_file_path.is_file()):
                    return True
                else:
                    return False
            case Language.TypeScript:
                current_dir = os.path.dirname(current_file_path)
                importee_file_path = os.path.abspath(os.path.join(current_dir, f"{package_name}.ets"))
                exist = os.path.exists(importee_file_path)
                if exist:
                    return False
                else:
                    return True
            case Language.CSharp:
                key_name = package_name
                if key_name not in module_map.keys():
                    return True
                else:
                    return False
            case _:
                raise NotImplementedError

    @staticmethod
    def find_import_node(graph, package_name, api_name, language, module_map, current_file_path):
        match language:
            case Language.Java:
                key_name = f"{package_name}.{api_name}"
                if key_name not in module_map.keys():
                    return []
                importee_file_path = module_map[key_name][0]
                node_found = graph.get_nodes(
                    node_filter=lambda x: (x.location and x.location and str(x.location.file_path) == str(importee_file_path))
                                          and x.name.strip() == api_name.strip() + ".java"
                )
                if len(node_found) > 0:
                    return node_found
                else:
                    return []
            case Language.TypeScript:
                current_dir = os.path.dirname(current_file_path)
                importee_file_path = os.path.abspath(os.path.join(current_dir, f"{package_name}.ets"))
                node_found = graph.get_nodes(
                    node_filter=lambda x: x.location and (str(x.location.file_path) == importee_file_path) and x.type==NodeType.FILE
                )
                if len(node_found) > 0:
                    return node_found
                else:
                    return []
            case Language.CSharp:
                key_name = f"{package_name}"
                if key_name not in module_map.keys():
                    return []
                importee_file_path = [str(f) for f in module_map[key_name]]
                node_found = graph.get_nodes(
                    node_filter=lambda x: (x.location and x.location and str(x.location.file_path) in importee_file_path
                                           and x.type == NodeType.FILE)
                )
                if len(node_found) > 0:
                    return node_found
                else:
                    return []
            case _:
                raise NotImplementedError


    @staticmethod
    def generate_single_file_package(graph, package_map, name):
        new_edges = []
        paths = package_map[name]
        package_node = Node(type=NodeType.PACKAGE,
                            name=name.strip(),
                            location=None,
                            content=name.strip())
        if not graph.graph.has_node(package_node):
            graph.add_node(package_node)
        for path in paths:
            new_edges.append((package_node,
                              path,
                              Edge(relation=EdgeRelation.HasMember,
                                   location=None)
                            ))
        return new_edges

    @staticmethod
    def resolving_imports(repo, graph, module_map, import_map, current_file_node):
        new_edges = []
        import_symbol_nodes = import_map[current_file_node]
        for importee_node in import_symbol_nodes:
            package_name, api_name_list = importee_node[0], importee_node[1]
            for api_name in api_name_list:
                node_list = TreeSitterDependencyGraphGenerator.find_import_node(graph, package_name,
                                                                                api_name, repo.language, module_map,
                                                                                str(current_file_node.location.file_path))
                for v in node_list:
                    if not graph.graph.has_edge(v, current_file_node):
                        new_edges.append(((v, current_file_node,
                                                  Edge(
                                                      relation=EdgeRelation.ImportedBy,
                                                      location=None,
                                                  )
                                                  )))
        return

    @staticmethod
    def generate_file_level_dependencies(graph: DependencyGraph, repo: Repository):
        repo_node = graph.get_nodes(node_filter=lambda x: x.type == NodeType.REPO)[0]
        module_map: dict[str, list[Node]] = defaultdict(list)
        import_map: dict[Node, list[TS_Node]] = defaultdict(list)

        finder = ImportFinder(repo.language)
        all_file_nodes = graph.get_nodes(node_filter=lambda x: x.type == NodeType.FILE)
        for file_node in tqdm(all_file_nodes, desc="Generating graph"):
            content = file_node.location.get_text()
            if not content.strip():
                continue
            if name := finder.find_module_name(content, Path(file_node.location.file_path)):
                module_map[name].append(file_node)
            nodes = finder.find_imports(content)
            import_map[file_node].extend(nodes)  # 当前文件，当前package import了这些节点

        max_processes = os.cpu_count()
        if repo.language in [Language.Java, Language.CSharp]:
            items = list(module_map.keys())
            worker_partial = partial(TreeSitterDependencyGraphGenerator.generate_single_file_package,
                                     graph, module_map)
            with ProcessPoolExecutor(max_processes) as executor:
                all_edges = list(tqdm(executor.map(worker_partial, items), total=len(items), desc="Adding package"))
            for edges in all_edges:
                if edges:
                    graph.add_relational_edges_from(edges)

        api_items = list(import_map.keys())
        worker_partial = partial(TreeSitterDependencyGraphGenerator.resolving_imports, repo,
                                 graph, module_map, import_map)
        with ProcessPoolExecutor(max_processes) as executor:
            all_edges = list(tqdm(executor.map(worker_partial, api_items), total=len(api_items), desc="Resolving imports"))
            for edges in all_edges:
                if edges:
                    graph.add_relational_edges_from(all_edges)
        return graph

    @staticmethod
    def generate_single_class(graph, all_class_name, class_node):
        new_edges = []
        file_node = graph.get_membership_parent(class_node)
        for extend_name in class_node.attribute['extends']:
            candidate_nodes = graph.get_nodes(
                node_filter=lambda node: node.type == NodeType.CLASS and node.name == extend_name)
            if len(candidate_nodes) == 0:
                continue
            for candidate_node in candidate_nodes:
                candidate_node_file = graph.get_membership_parent(candidate_node)
                if file_node == candidate_node_file or graph.graph.has_edge(candidate_node_file, file_node):
                    new_edges.append((candidate_node, class_node,
                                              Edge(
                                                  relation=EdgeRelation.BaseClassOf,
                                                  location=None
                                              )))
                else:
                    file_package_list = [n for n in graph.graph.predecessors(file_node)
                                         if
                                         graph.graph[n][file_node][0]['relation'].relation == EdgeRelation.HasMember
                                         and n.type == NodeType.PACKAGE]
                    candidate_file_package_list = [n for n in graph.graph.predecessors(candidate_node_file)
                                                   if graph.graph[n][candidate_node_file][0][
                                                       'relation'].relation == EdgeRelation.HasMember
                                                   and n.type == NodeType.PACKAGE]
                    if len(file_package_list) > 0 and len(candidate_file_package_list) > 0:
                        if file_package_list[0] == candidate_file_package_list[0]:
                            new_edges.append((candidate_node, class_node,
                                                      Edge(
                                                          relation=EdgeRelation.BaseClassOf,
                                                          location=None
                                                      )))
        for implement_name in class_node.attribute['implements']:
            candidate_nodes = graph.get_nodes(
                node_filter=lambda node: node.type == NodeType.INTERFACE and node.name == implement_name)
            if len(candidate_nodes) == 0:
                continue
            for candidate_node in candidate_nodes:
                candidate_node_file = graph.get_membership_parent(candidate_node)
                if file_node == candidate_node_file or graph.graph.has_edge(candidate_node_file, file_node):
                    new_edges.append((candidate_node, class_node,
                                              Edge(
                                                  relation=EdgeRelation.ImplementedBy,
                                                  location=None
                                              )))
                else:
                    file_package_list = [n for n in graph.graph.predecessors(file_node)
                                         if
                                         graph.graph[n][file_node][0]['relation'].relation == EdgeRelation.HasMember
                                         and n.type == NodeType.PACKAGE]
                    candidate_file_package_list = [n for n in graph.graph.predecessors(candidate_node_file)
                                                   if graph.graph[n][candidate_node_file][0][
                                                       'relation'].relation == EdgeRelation.HasMember
                                                   and n.type == NodeType.PACKAGE]
                    if len(file_package_list) > 0 and len(candidate_file_package_list) > 0:
                        if file_package_list[0] == candidate_file_package_list[0]:
                            new_edges.append((candidate_node, class_node,
                                                      Edge(
                                                          relation=EdgeRelation.ImplementedBy,
                                                          location=None
                                                      )))
        for use_name in class_node.attribute['uses']:
            if use_name not in all_class_name:
                continue
            candidate_nodes = graph.get_nodes(
                node_filter=lambda node: str(node.type) == NodeType.CLASS and node.name == use_name)
            if len(candidate_nodes) == 0:
                continue
            for candidate_node in candidate_nodes:
                candidate_node_file = graph.get_membership_parent(candidate_node)
                if file_node == candidate_node_file or graph.graph.has_edge(candidate_node_file, file_node):
                    if not graph.graph.has_edge(candidate_node, class_node):
                        new_edges.append((candidate_node, class_node,
                                                  Edge(
                                                      relation=EdgeRelation.UsedBy,
                                                      location=None
                                                  )))
                else:
                    file_package_list = [n for n in graph.graph.predecessors(file_node)
                                         if
                                         graph.graph[n][file_node][0]['relation'].relation == EdgeRelation.HasMember
                                         and n.type == NodeType.PACKAGE]
                    candidate_file_package_list = [n for n in graph.graph.predecessors(candidate_node_file)
                                                   if graph.graph[n][candidate_node_file][0][
                                                       'relation'].relation == EdgeRelation.HasMember
                                                   and n.type == NodeType.PACKAGE]
                    if len(file_package_list) > 0 and len(candidate_file_package_list) > 0:
                        if file_package_list[0] == candidate_file_package_list[0]:
                            if not graph.graph.has_edge(candidate_node, class_node):
                                new_edges.append((candidate_node, class_node,
                                                  Edge(
                                                      relation=EdgeRelation.UsedBy,
                                                      location=None
                                                  )))
        return new_edges

    @staticmethod
    def generate_class_level_dependencies(graph: DependencyGraph, repo: Repository):

        all_class_nodes = graph.get_nodes(node_filter=lambda node: node.type == NodeType.CLASS)
        all_class_name = [node.name for node in all_class_nodes]
        max_processes = os.cpu_count()
        worker_partial = partial(TreeSitterDependencyGraphGenerator.generate_single_class, graph, all_class_name)
        with ProcessPoolExecutor(max_processes) as executor:
            all_edges = list(tqdm(executor.map(worker_partial, all_class_nodes),
                                  total=len(all_class_nodes), desc="Adding class level dependencies"))
            for edges in all_edges:
                if edges:
                    graph.add_relational_edges_from(edges)
        return graph

    @staticmethod
    def generate_single_method(graph, method_node):
        new_edges = []
        class_node = graph.get_membership_parent(method_node)
        all_method_call = set(method_node.attribute['method_call_list'])
        for name in list(all_method_call):
            candidate_nodes = graph.get_nodes(
                node_filter=lambda node: node.type == NodeType.METHOD and node.name == name.strip("("))
            candidate_nodes.sort(key=lambda x: x.name)
            if len(candidate_nodes) == 0:
                continue
            for candidate_node in candidate_nodes:
                candidate_node_class = graph.get_membership_parent(candidate_node)
                if candidate_node_class == class_node or graph.graph.has_edge(candidate_node_class, class_node):
                    if not graph.graph.has_edge(candidate_node, method_node):
                        new_edges.append((candidate_node, method_node,
                                          Edge(
                                              relation=EdgeRelation.UsedBy,
                                              location=None
                                          )))
        method_body = method_node.location.get_text()
        successors = list(graph.graph.successors(class_node))
        successors.sort(key=lambda x: x.name)
        for node in successors:
            if node.type == NodeType.FIELD:
                clean_name = node.name.strip().strip(';')
                if clean_name not in method_body:
                    continue
                parent_field = graph.get_membership_parent(node)
                if parent_field == class_node:
                    new_edges.append((node, method_node,
                                              Edge(
                                                  relation=EdgeRelation.UsedBy,
                                                  location=None
                                              )))
                elif graph.get_membership_parent(class_node):  # 内部类
                    pp = graph.get_membership_parent(class_node)
                    if parent_field == pp:
                        new_edges.append((node, method_node,
                                                  Edge(
                                                      relation=EdgeRelation.UsedBy,
                                                      location=None
                                                  )))
        file_node = graph.get_membership_parent(class_node)
        import_nodes = [source for source, target, data in graph.graph.in_edges(file_node, data=True) if
                        data['relation'].relation == EdgeRelation.ImportedBy and source.type == NodeType.VIRTUAL_API]
        for v in import_nodes:
            if v.name in method_body:
                if not graph.graph.has_edge(v, method_node):
                    new_edges.append((v, method_node,
                                              Edge(
                                                  relation=EdgeRelation.UsedBy,
                                                  location=None
                                              )))
        return new_edges

    @staticmethod
    def generate_method_level_dependencies(graph: DependencyGraph, repo: Repository):

        all_method_nodes = graph.get_nodes(
            node_filter=lambda node: node.type == NodeType.METHOD or node.type == NodeType.CONSTRUCTOR)
        all_method_nodes.sort(key=lambda x: x.name)
        max_processes = os.cpu_count()
        worker_partial = partial(TreeSitterDependencyGraphGenerator.generate_single_method, graph)
        with ProcessPoolExecutor(max_processes) as executor:
            all_edges = list(tqdm(executor.map(worker_partial, all_method_nodes), total=len(all_method_nodes),
                      desc="Adding method level dependencies"))
            for edges in all_edges:
                if edges:
                    graph.add_relational_edges_from(edges)
        return graph