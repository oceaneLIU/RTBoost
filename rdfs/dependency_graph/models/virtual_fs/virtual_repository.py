from collections import namedtuple

from fs.memoryfs import MemoryFS

from rdfs.dependency_graph.models import PathLike
from rdfs.dependency_graph.models.language import Language
from rdfs.dependency_graph.models.repository import Repository
from rdfs.dependency_graph.models.virtual_fs.virtual_file_node import VirtualFileNode
from rdfs.dependency_graph.models.virtual_fs.virtual_path import VirtualPath

# Define the VirtualFile named tuple
VirtualFile = namedtuple("VirtualFile", ["relative_path", "content"])


class VirtualRepository(Repository):
    def __init__(
        self,
        repo_path: PathLike,
        language: Language,
        virtual_files: list[VirtualFile],  # Use the named tuple for typing
    ):
        self.fs = MemoryFS()
        # Make sure the repo path is absolute
        self.repo_path = VirtualPath(self.fs, "/", repo_path)
        self.repo_path.mkdir(parents=True)

        self._all_file_paths = []
        for file_path, content in virtual_files:
            # Strip the leading slash on the file path
            p = VirtualPath(self.fs, self.repo_path / file_path.lstrip("/"))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            self._all_file_paths.append(p)

        super().__init__(self.repo_path, language)

    @property
    def files(self) -> set[VirtualFileNode]:
        files: set[VirtualFileNode] = set()
        for file_path in self._all_file_paths:
            if file_path.suffix in self.code_file_extensions[self.language]:
                files.add(VirtualFileNode(file_path))
        return files
