import enum
from dataclasses import dataclass, field
from pathlib import Path
import json
from dataclasses_json import dataclass_json, config

from rdfs.dependency_graph.utils.log import setup_logger
from rdfs.dependency_graph.models.language import Language
from rdfs.dependency_graph.utils.mypy_stub import generate_python_stub
from rdfs.dependency_graph.utils.text import slice_text
from typing import Optional

# Initialize logging
logger = setup_logger()


class EdgeRelation(enum.Enum):
    """The relation between two nodes"""

    ImportedBy = "imports"
    BaseClassOf = "has base class"
    UsedBy = "uses"
    HasMember = "is member of"
    ImplementedBy = "implements"
    Imports = "imports"

    def __str__(self):
        return self.name

    @classmethod
    def __getitem__(cls, name):
        return cls[name]

    def get_inverse_kind(self) -> "EdgeRelation":
        new_value = [*self.value]
        new_value[2] = 1 - new_value[2]
        return EdgeRelation(tuple(new_value))

    def is_inverse_relationship(self, other) -> bool:
        return (
            self.value[0] == other.value[0]
            and self.value[1] == other.value[1]
            and self.value[2] != other.value[2]
        )


@dataclass_json
@dataclass
class Location:
    def __str__(self) -> str:
        signature = f"{self.file_path}"
        loc = [self.start_line, self.start_column, self.end_line, self.end_column]
        if any([l is not None for l in loc]):
            signature += f":{self.start_line}:{self.start_column}-{self.end_line}:{self.end_column}"

        return signature

    def __hash__(self) -> int:
        return hash(self.__str__())

    def get_text(self) -> Optional[str]:
        # TODO should leverage the FileNode.content
        if self.file_path is None:
            return None
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        loc = [self.start_line, self.start_column, self.end_line, self.end_column]
        if any([l is None for l in loc]):
            return content
        return slice_text(
            content, self.start_line, self.start_column, self.end_line, self.end_column
        )

    file_path: Optional[Path] = field(
        default=None,
        metadata=config(encoder=lambda v: str(v), decoder=lambda v: Path(v)),
    )
    """The file path"""
    start_line: Optional[int] = None
    """The start line number, 1-based"""
    start_column: Optional[int] = None
    """The start column number, 1-based"""
    end_line: Optional[int] = None
    """The end line number, 1-based"""
    end_column: Optional[int] = None
    """The end column number, 1-based"""


class NodeType(str, enum.Enum):
    # TODO should nest a language to mark different type for different language
    PACKAGE = "package"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    VARIABLE = "variable"
    STATEMENT = "statement"
    FILE = "file"
    IMPORT = "import"
    REPO = "repository"
    DIR = "directory"
    INTERFACE = "interface"
    ENUM = "enum class"
    FIELD = "class field"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    VIRTUAl_PACKAGE = "third-party package"
    VIRTUAL_API = "third-party api"
    def __str__(self):
        return self.value


@dataclass_json
@dataclass
class Node:
    def __str__(self) -> str:
        return f"{self.name}@:{self.type.value}@:{self.location}@:{json.dumps(self.attribute)}@:{self.content}"

    def __hash__(self) -> int:
        return hash(self.__str__())

    def get_text(self) -> Optional[str]:
        return self.location.get_text()

    def get_stub(
        self, language: Language, include_comments: bool = False
    ) -> Optional[str]:
        if language == Language.Python:
            return generate_python_stub(self.get_text())
        elif language == Language.Java:
            from repo_specific_semantic_graph.dependency_graph.utils.tree_sitter_stub import generate_java_stub

            return generate_java_stub(
                self.get_text(), include_comments=include_comments
            )
        elif language == Language.CSharp:
            from repo_specific_semantic_graph.dependency_graph.utils.tree_sitter_stub import (
                generate_c_sharp_stub,
            )

            return generate_c_sharp_stub(
                self.get_text(), include_comments=include_comments
            )
        elif language == Language.TypeScript | Language.JavaScript:
            from repo_specific_semantic_graph.dependency_graph.utils.tree_sitter_stub import (
                generate_ts_js_stub,
            )

            return generate_ts_js_stub(
                self.get_text(), include_comments=include_comments
            )
        else:
            logger.warning(f"Stub generation is not supported for {language}")
            return None
    """The type of the node"""
    type: NodeType = field(
        metadata=config(
            encoder=lambda v: NodeType(v).value, decoder=lambda v: NodeType(v)
        )
    )
    """The name of the node"""
    name: str
    """The location of the node"""
    location: Location
    """The attributes of the node"""
    attribute: dict = None
    """The content of the node to be translated"""
    content: str = ""

@dataclass_json
@dataclass
class Edge:
    def __str__(self) -> str:
        signature = f"{self.relation}"
        if self.location:
            signature += f"@{self.location}"
        return signature

    def __hash__(self) -> int:
        return hash(self.__str__())

    def get_text(self) -> Optional[str]:
        return self.location.get_text()

    def get_inverse_edge(self) -> "Edge":
        return Edge(
            relation=self.relation.get_inverse_kind(),
            location=self.location,
        )

    relation: EdgeRelation = field(
        metadata=config(encoder=lambda v: str(v), decoder=lambda v: EdgeRelation[v])
    )
    """The relation between two nodes"""
    location: Optional[Location] = None
    """The location of the edge"""
