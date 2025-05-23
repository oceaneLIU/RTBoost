import os
from pathlib import Path

from importlab.parsepy import ImportStatement
from importlab.resolve import ImportException
from tree_sitter import Node as TS_Node

from rdfs.dependency_graph.models import VirtualPath, PathLike
from rdfs.dependency_graph.models.language import Language
from rdfs.dependency_graph.models.repository import Repository
from rdfs.dependency_graph.utils.log import setup_logger

# Initialize logging
logger = setup_logger()


class ImportResolver:
    def __init__(self, repo: Repository):
        self.repo = repo

    def _Path(self, file_path: PathLike) -> Path:
        """
        Convert the str file path to handle both physical and virtual paths
        """
        match self.repo.repo_path:
            case Path():
                return Path(file_path)
            case VirtualPath():
                return VirtualPath(self.repo.repo_path.fs, file_path)
            case _:
                return Path(file_path)

    def resolve_import(
        self,
        import_symbol_node: TS_Node,
        module_map: dict[str, list[Path]],
        importer_file_path: Path,
    ) -> list[Path]:
        resolved_path_list = []

        match self.repo.language:
            case Language.Java | Language.Kotlin:
                import_symbol_name = import_symbol_node.text.decode()
                # Deal with star import: `import xxx.*`
                if b".*" in import_symbol_node.parent.text:
                    for module_name, path_list in module_map.items():
                        # Use rpartition to split the string at the rightmost '.'
                        package_name, _, _ = module_name.rpartition(".")
                        if package_name == import_symbol_name:
                            resolved_path_list.extend(path_list)
                else:
                    resolved_path_list.extend(module_map.get(import_symbol_name, []))
            case Language.CSharp:
                import_symbol_name = import_symbol_node.text.decode()
                resolved_path_list.extend(module_map.get(import_symbol_name, []))
            case Language.TypeScript | Language.JavaScript:
                resolved_path_list.extend(
                    self.resolve_ts_js_import(
                        import_symbol_node, module_map, importer_file_path
                    )
                )
            case Language.Python:
                resolved_path_list.extend(
                    self.resolve_python_import(import_symbol_node, importer_file_path)
                )
            case Language.PHP:
                resolved_path_list.extend(
                    self.resolve_php_import(import_symbol_node, importer_file_path)
                )
            case Language.Ruby:
                resolved_path_list.extend(
                    self.resolve_ruby_import(import_symbol_node, importer_file_path)
                )
            case Language.C | Language.CPP:
                resolved_path_list.extend(
                    self.resolve_cfamily_import(import_symbol_node, importer_file_path)
                )
            case Language.Go:
                resolved_path_list.extend(self.resolve_go_import(import_symbol_node))
            case Language.Swift:
                resolved_path_list.extend(
                    self.resolve_swift_import(import_symbol_node, importer_file_path)
                )
            case Language.Rust:
                resolved_path_list.extend(
                    self.resolve_rust_import(import_symbol_node, importer_file_path)
                )
            case Language.Lua:
                resolved_path_list.extend(
                    self.resolve_lua_import(import_symbol_node, importer_file_path)
                )
            case Language.Bash:
                resolved_path_list.extend(
                    self.resolve_bash_import(import_symbol_node, importer_file_path)
                )
            case Language.R:
                resolved_path_list.extend(
                    self.resolve_r_import(import_symbol_node, importer_file_path)
                )
            case _:
                raise NotImplementedError(
                    f"Language {self.repo.language} is not supported"
                )

        # De-duplicate the resolved path
        return list(set(resolved_path_list))

    def resolve_ts_js_import(
        self,
        import_symbol_node: TS_Node,
        module_map: dict[str, list[Path]],
        importer_file_path: Path,
    ) -> list[Path]:
        def _search_file(search_path: Path, module_name: str) -> list[Path]:
            result_path = []
            for ext in extension_list:
                if (search_path / f"{module_name}{ext}").exists():
                    result_path.append(search_path / f"{module_name}{ext}")
                elif (search_path / f"{module_name}").is_dir():
                    """
                    In case the module is a directory, we should search for the `module_dir/index.{js|ts}` file
                    """
                    for ext in extension_list:
                        if (search_path / f"{module_name}" / f"index{ext}").exists():
                            result_path.append(
                                search_path / f"{module_name}" / f"index{ext}"
                            )
                    break
            return result_path

        import_symbol_name = import_symbol_node.text.decode()
        extension_list = (
            Repository.code_file_extensions[Language.TypeScript]
            + Repository.code_file_extensions[Language.JavaScript]
        )

        # Find the module path
        # e.g. './Descriptor' -> './Descriptor.ts'; '../Descriptor' -> '../Descriptor.ts'
        if "." in import_symbol_name or ".." in import_symbol_name:
            result_path = []
            # If there is a suffix in the name
            if suffix := self._Path(import_symbol_name).suffix:
                # In case of '../package.json', we should filter it out
                path = importer_file_path.parent / import_symbol_name
                if suffix in extension_list and path.exists():
                    result_path = [path]
            else:
                result_path = _search_file(
                    importer_file_path.parent, import_symbol_name
                )
            return result_path
        else:
            return module_map.get(import_symbol_name, [])


    def resolve_php_import(
        self,
        import_symbol_node: TS_Node,
        importer_file_path: Path,
    ) -> list[Path]:
        import_symbol_name = import_symbol_node.text.decode()
        # Strip double and single quote
        import_symbol_name = import_symbol_name.strip('"').strip("'")
        # Find the module path
        result_path = []
        import_path = self._Path(import_symbol_name)
        if import_path.is_absolute() and import_path.exists():
            result_path.append(import_path)
        else:
            path = importer_file_path.parent / import_symbol_name
            if path.exists():
                result_path.append(path)
        return result_path

    def resolve_ruby_import(
        self,
        import_symbol_node: TS_Node,
        importer_file_path: Path,
    ) -> list[Path]:
        import_symbol_name = import_symbol_node.text.decode()
        # Strip double and single quote
        import_symbol_name = import_symbol_name.strip('"').strip("'")

        extension_list = Repository.code_file_extensions[Language.Ruby]

        # Find the module path
        result_path = []
        for ext in extension_list:
            try:
                import_path = self._Path(import_symbol_name).with_suffix(ext)
            except ValueError:
                continue

            if import_path.is_absolute() and import_path.exists():
                result_path.append(import_path)
            else:
                path = importer_file_path.parent / import_symbol_name
                path = path.with_suffix(ext)
                if path.exists():
                    result_path.append(path)

        return result_path

    def resolve_cfamily_import(
        self,
        import_symbol_node: TS_Node,
        importer_file_path: Path,
    ) -> list[Path]:

        import_symbol_name = import_symbol_node.text.decode()
        # Strip double quote and angle bracket
        import_symbol_name = import_symbol_name.strip('"').lstrip("<").rstrip(">")
        import_path = self._Path(import_symbol_name)

        # Heuristics to search for the header file
        search_paths = [
            # Common practice to have headers in 'include' directory
            self.repo.repo_path / "include" / import_path,
            # Relative path from the C file's directory
            importer_file_path.parent / import_path,
            # Common practice to have headers in 'src' directory
            self.repo.repo_path / "src" / import_path,
            # Absolute/relative path as given in the include statement
            import_path,
        ]

        # Add parent directories of the C file path
        for parent in importer_file_path.parents:
            search_paths.append(parent / import_path)

        # Add sibling directories of each directory component of importer_file_path
        for parent in importer_file_path.parents:
            for sibling in parent.iterdir():
                if sibling.is_dir() and sibling != importer_file_path:
                    search_paths.append(sibling / import_path)

        # Find the module path
        result_path = []
        # Check if any of these paths exist
        for path in search_paths:
            if path.exists():
                result_path.append(path)

        return result_path

    def resolve_go_import(self, import_symbol_node: TS_Node) -> list[Path]:
        def parse_go_mod(go_mod_path: Path) -> tuple[str, dict[str, Path]]:
            """
            Parses the go.mod file and returns the module path and replacements.
            :param go_mod_path: The path to the go.mod file.
            :return: A tuple containing the module path and replacements.
            """
            module_path = None
            replacements = {}

            for line in go_mod_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("module "):
                    module_path = line.split()[1]
                elif line.startswith("replace "):
                    parts = line.split()
                    if len(parts) >= 4 and parts[2] == "=>":
                        replacements[parts[1]] = self._Path(parts[3])

            return module_path, replacements

        def search_fallback_paths(import_stmt: str, base_path: Path):
            """Searches various fallback paths within the project directory."""
            search_paths = [
                base_path / import_stmt.replace("/", os.sep),
                base_path / "src" / import_stmt.replace("/", os.sep),
                base_path / "vendor" / import_stmt.replace("/", os.sep),
                base_path / "pkg" / import_stmt.replace("/", os.sep),
            ]
            found_files = []

            for path in search_paths:
                if path.is_dir():
                    go_files = list(path.glob("*.go"))
                    if go_files:
                        found_files.extend(go_files)
                elif path.with_suffix(".go").is_file():
                    found_files.append(path.with_suffix(".go"))

            return found_files

        # Parse the go.mod file
        go_mod_path = self.repo.repo_path / "go.mod"
        if go_mod_path.exists():
            module_path, replacements = parse_go_mod(go_mod_path)
        else:
            module_path, replacements = None, {}

        # Find corresponding paths for the imported packages
        imported_paths = []

        import_stmt = import_symbol_node.text.decode()
        import_stmt = import_stmt.strip('"')

        # Resolve the import path using replacements or the module path
        resolved_paths = []
        if import_stmt in replacements:
            resolved_path = replacements[import_stmt]
            resolved_paths.append(resolved_path)
        elif module_path and import_stmt.startswith(module_path):
            resolved_path = self.repo.repo_path / import_stmt[len(module_path) + 1 :]
            resolved_paths.append(resolved_path)
        else:
            # Fallback logic: Try to resolve based on project directory structure
            resolved_paths.extend(
                search_fallback_paths(import_stmt, self.repo.repo_path)
            )

        for resolved_path in resolved_paths:
            if resolved_path and resolved_path.is_dir():
                # Try to find a .go file in the directory
                go_files = list(resolved_path.glob("*.go"))
                if go_files:
                    imported_paths.extend(go_files)

        return imported_paths

    def resolve_swift_import(
        self, import_symbol_node: TS_Node, importer_file_path: Path
    ) -> list[Path]:
        import_symbol_name = import_symbol_node.text.decode()
        if len(import_symbol_node.parent.children) > 2:
            # Handle individual declarations importing such as `import kind module.symbol`
            # In this case, we extract the module name from the import statement
            import_symbol_name = (
                ".".join(import_symbol_name.split(".")[:-1])
                if "." in import_symbol_name
                else import_symbol_name
            )

        import_symbol_name = import_symbol_name.replace(".", os.sep)
        import_path = self._Path(import_symbol_name)

        # Heuristic search for source files corresponding to the imported modules
        search_paths = [
            self.repo.repo_path / "Sources" / import_symbol_name,
            self.repo.repo_path / "Tests" / import_symbol_name,
            self.repo.repo_path / "Modules" / import_symbol_name,
        ]

        # Add parent directories of the Swift file path
        for parent in importer_file_path.parents:
            search_paths.append(parent / import_path)

        # Add sibling directories of each directory component of importer_file_path
        for parent in importer_file_path.parents:
            for sibling in parent.iterdir():
                if sibling.is_dir() and sibling != importer_file_path:
                    search_paths.append(sibling / import_path)

        # Heuristic search for source files corresponding to the imported modules
        result_files = []

        for path in search_paths:
            extension_list = Repository.code_file_extensions[Language.Swift]
            if path.exists() and path.is_dir():
                for ext in extension_list:
                    for swift_file in path.glob(f"**/*{ext}"):
                        result_files.append(swift_file)

        # Return list of Path objects corresponding to the imported files
        return result_files

    def resolve_rust_import(
        self, import_symbol_node: TS_Node, importer_file_path: Path
    ) -> list[Path]:
        def find_import_path(
            project_root: Path, file: Path, module_path: list[str], is_absolute: bool
        ) -> Path | None:
            """
            Given the project root, the file containing the import, and the module path,
            heuristically find the corresponding file path for the imported module.

            :param project_root: The root directory of the Rust project.
            :param file: The file (pathlib.Path) containing the import statement.
            :param module_path: A list of module components (e.g., ["my_module", "sub_module"]).
            :param is_absolute: Boolean indicating if the import is absolute (`crate::`).
            :return: The pathlib.Path object for the corresponding file or None if not found.
            """
            # Start from the project root if the path is absolute
            current_dir = project_root / "src" if is_absolute else file.parent

            for part in module_path:
                # Check if the module is a directory with a mod.rs or a file <module_name>.rs
                dir_path = current_dir / part
                mod_file_path = dir_path / "mod.rs"
                file_path = current_dir / f"{part}.rs"

                if mod_file_path.exists():
                    current_dir = dir_path
                elif file_path.exists():
                    return file_path
                else:
                    # If not found, check further up the directory hierarchy for relative imports
                    if not is_absolute:
                        found = False
                        for ancestor in current_dir.parents:
                            ancestor_dir_path = ancestor / part
                            ancestor_mod_file_path = ancestor_dir_path / "mod.rs"
                            ancestor_file_path = ancestor / f"{part}.rs"

                            if ancestor_mod_file_path.exists():
                                current_dir = ancestor_dir_path
                                found = True
                                break
                            elif ancestor_file_path.exists():
                                return ancestor_file_path

                        if not found:
                            return None
                    else:
                        return None

            # If we reach here, assume the last module part is a file
            final_file = current_dir / f"{module_path[-1]}.rs"
            if final_file.exists():
                return final_file
            else:
                return None

        # Decode the symbol name and split into module path components
        import_symbol_name = import_symbol_node.text.decode()
        module_path = import_symbol_name.split("::")

        # Determine if the import is absolute
        is_absolute = module_path[0] == "crate"
        if is_absolute:
            module_path = module_path[1:]  # Remove the leading "crate"

        # Attempt to find the imported file based on heuristics
        imported_file = find_import_path(
            self.repo.repo_path, importer_file_path, module_path, is_absolute
        )
        return [imported_file] if imported_file else []

    def resolve_lua_import(
        self, import_symbol_node: TS_Node, importer_file_path: Path
    ) -> list[Path]:
        import_symbol_name = import_symbol_node.text.decode()
        import_symbol_name = import_symbol_name.strip('"').strip("'")

        extension_list = Repository.code_file_extensions[Language.Lua]

        # Here, we make sure in case of `dofile("module4.lua")`, the `.lua` suffix is preserved.
        if all(ext not in import_symbol_name for ext in extension_list):
            """
            In case of `require("submodule.module2")`, the Lua file is expected to be located at
            `submodule/module2.lua`. But before replacing the `.` with `/`, we need to make sure in case of
            `dofile("module4.lua")`, the `.lua` suffix is preserved.
            """
            import_symbol_name = import_symbol_name.replace(".", os.sep)

        resolved_path = importer_file_path.parent / import_symbol_name

        if resolved_path.exists():
            return [resolved_path]

        for ext in extension_list:
            path = resolved_path.with_suffix(ext)
            if path.exists():
                return [resolved_path.with_suffix(ext)]
        return []

    def resolve_bash_import(
        self, import_symbol_node: TS_Node, importer_file_path: Path
    ) -> list[Path]:
        import_symbol_name = import_symbol_node.text.decode()
        if self._Path(import_symbol_name).exists():
            return [self._Path(import_symbol_name)]
        else:
            resolved_path = importer_file_path.parent / import_symbol_name
            if resolved_path.exists():
                return [resolved_path]
        return []

    def resolve_r_import(
        self, import_symbol_node: TS_Node, importer_file_path: Path
    ) -> list[Path]:
        import_symbol_name = import_symbol_node.text.decode()
        import_symbol_name = import_symbol_name.strip('"').strip("'")

        if self._Path(import_symbol_name).exists():
            return [self._Path(import_symbol_name)]
        else:
            resolved_path = importer_file_path.parent / import_symbol_name
            if resolved_path.exists():
                return [resolved_path]
        return []
