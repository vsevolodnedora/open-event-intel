from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

import yaml
from jinja2 import Environment
from jinja2.loaders import FileSystemLoader


@dataclass(slots=True)
class PromptMetadata:
    """Prompt metadata."""

    name: str | None = None
    version: str | None = None
    author: str | None = None
    date: str | None = None
    notes: str | None = None

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PromptMetadata":
        """Create a PromptMetadata from a dictionary."""
        if data is None:
            data = {}
        return cls(
            name=str(data["name"]) if "name" in data else None,
            version=str(data["version"]) if "version" in data else None,
            raw=dict(data),
        )


def _split_front_matter(source: str) -> tuple[dict[str, Any], str]:
    lines = source.splitlines(keepends=True)
    if not lines:
        return {}, source

    start = 0
    while start < len(lines) and lines[start].strip() == "":
        start += 1

    if start >= len(lines) or lines[start].strip() != "---METADATA---":
        return {}, source

    end = start + 1
    while end < len(lines) and lines[end].strip() != "---METADATA---":
        end += 1

    if end >= len(lines):
        return {}, source

    yaml_block = "".join(lines[start + 1 : end])
    body = "".join(lines[end + 1 :])

    meta_raw = yaml.safe_load(yaml_block) or {}
    if not isinstance(meta_raw, dict):
        meta_raw = {}

    return meta_raw, body


class FrontMatterFileSystemLoader(FileSystemLoader):
    """Extension of the FileSystemLoader with front matter."""

    def __init__(self, searchpath: str | Path, encoding: str = "utf-8") -> None:
        """Initialize a FrontMatterFileSystemLoader."""
        super().__init__(searchpath=str(searchpath), encoding=encoding)
        self._metadata_cache: MutableMapping[str, dict[str, Any]] = {}

    def get_source(self, environment: Environment, template: str):
        """Get the source code for the given template."""
        source, filename, uptodate = super().get_source(environment, template)
        meta_dict, body = _split_front_matter(source)
        self._metadata_cache[template] = meta_dict
        return body, filename, uptodate


class PromptRegistry:
    """
    Thin registry for prompts + associated private data.

    - Jinja templates with YAML front matter
    - Arbitrary YAML data files (like LABEL_DEFINITIONS)
    """

    def __init__(self, prompts_path: str | Path) -> None:
        """Initialize a PromptRegistry."""
        prompts_path = Path(prompts_path)
        if not prompts_path.is_dir():
            raise FileNotFoundError(f"Prompts directory not found: {prompts_path}")

        self._root = prompts_path
        self._loader = FrontMatterFileSystemLoader(prompts_path)
        self._meta_env = Environment(loader=self._loader, autoescape=False)

    def create_environment(
            self,
            filters: Mapping[str, Callable[..., Any]] | None = None,
    ) -> Environment:
        """Create a jinja environment."""
        env = Environment(loader=self._loader, autoescape=False)
        if filters:
            env.filters.update(filters)
        return env

    def get_metadata(self, template_name: str) -> PromptMetadata:
        """Get a PromptMetadata by name."""
        if template_name not in self._loader._metadata_cache:
            self._loader.get_source(self._meta_env, template_name)
        raw_meta = self._loader._metadata_cache.get(template_name, {}) or {}
        return PromptMetadata.from_dict(raw_meta)

    def render(
            self,
            template_name: str,
            context: Mapping[str, Any],
            filters: Mapping[str, Callable[..., Any]] | None = None,
    ) -> str:
        """Render a PromptMetadata by name."""
        env = self.create_environment(filters=filters)
        template = env.get_template(template_name)
        return template.render(**context)

    # Generic YAML loader for associated private data
    def load_yaml_dict(self, filename: str) -> dict[str, Any]:
        """
        Load a YAML file under the prompts root and return it as a dict.

        Example:
            registry.load_yaml_dict("label_definitions.yaml")

        :param filename: Name of the YAML file relative to prompts_path
        :return: Dictionary loaded from YAML file

        """
        path = self._root / filename
        if not path.is_file():
            raise FileNotFoundError(f"YAML file not found: {path}")

        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}

        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping at top-level in {path}, got {type(data).__name__}")

        return data

    def validate_files(self, filenames: list[str]) -> None:
        """
        Validate that required files exist in the prompts directory.

        :param filenames: Names of files to validate
        :raises FileNotFoundError: If any required file is missing
        """
        missing_files = []
        for filename in filenames:
            path = self._root / filename
            if not path.is_file():
                missing_files.append(filename)

        if missing_files:
            raise FileNotFoundError(
                f"Required files not found in {self._root}: {', '.join(missing_files)}"
            )

    def file_exists(self, filename: str) -> bool:
        """
        Check if a file exists in the prompts directory.

        :param filename: Name of file to check
        :return: True if file exists, False otherwise
        """
        path = self._root / filename
        return path.is_file()

    def load_predicate_definitions(self, filename: str = "predicate_definitions.yaml") -> dict[str, str]:
        """
        Load predicate definitions from a YAML file.

        Returns a dictionary mapping predicate names to their descriptions,
        compatible with the original PREDICATE_DEFINITIONS format.

        :param filename: Name of the YAML file (default: "predicate_definitions.yaml")
        :return: Dictionary of predicate_name -> description

        Example:
            predicate_defs = registry.load_predicate_definitions()
            # Returns: {"IS_TYPE": "Entity type classification...", ...}

        """
        data = self.load_yaml_dict(filename)

        if "predicate_definitions" not in data:
            raise ValueError(
                f"Expected 'predicate_definitions' key in {filename}, "
                f"found keys: {list(data.keys())}"
            )

        return data["predicate_definitions"]

    def load_predicate_groups(
            self,
            filename: str = "predicate_definitions.yaml",
            legacy_format: bool = False
    ) -> list[list[str]] | list[dict[str, Any]]:
        """
        Load predicate groups from a YAML file.

        :param filename: Name of the YAML file (default: "predicate_definitions.yaml")
        :param legacy_format: If True, returns list of lists (old format).
                             If False, returns structured list with metadata (new format).
        :return: List of predicate groups

        Example (legacy_format=True):
            groups = registry.load_predicate_groups(legacy_format=True)
            # Returns: [["IS_TYPE", "LOCATED_IN", ...], ["PUBLISHED", ...], ...]

        Example (legacy_format=False):
            groups = registry.load_predicate_groups(legacy_format=False)
            # Returns: [
            #   {"name": "State Predicates", "predicates": ["IS_TYPE", ...]},
            #   {"name": "Regulatory Events", "predicates": ["PUBLISHED", ...]},
            #   ...
            # ]
        """
        data = self.load_yaml_dict(filename)

        if "predicate_groups" not in data:
            raise ValueError(
                f"Expected 'predicate_groups' key in {filename}, "
                f"found keys: {list(data.keys())}"
            )

        groups = data["predicate_groups"]

        if legacy_format:
            # Return just the list of predicates for each group
            return [group["predicates"] for group in groups]
        else:
            # Return the full structured format
            return groups