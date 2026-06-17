from __future__ import annotations

import shutil
from pathlib import Path

from docforge.core.constants import (
    PROJECTS_DIR,
    CACHE_DIR,
    VECTORS_DIR,
    TEMPLATES_DIR,
    OUTPUTS_DIR,
)

from docforge.core.errors import ProjectError
# from docforge.core.logger import get_logger

# log = get_logger(__name__)


class ProjectManager:

    SUBDIRECTORIES = (
        CACHE_DIR,
        VECTORS_DIR,
        TEMPLATES_DIR,
        OUTPUTS_DIR,
    )

    def create_project(self, project_name: str) -> Path:
        project_name = self._validate_name(project_name)

        project_path = PROJECTS_DIR / project_name

        try:
            for directory in self.SUBDIRECTORIES:
                (project_path / directory).mkdir(
                    parents=True,
                    exist_ok=True,
                )

            log.info(
                "project_created",
                project=project_name,
            )

            return project_path

        except OSError as e:
            raise ProjectError(
                f"Failed to create project '{project_name}': {e}"
            ) from e

    def delete_project(self, project_name: str) -> None:
        project_name = self._validate_name(project_name)

        project_path = PROJECTS_DIR / project_name

        if not project_path.exists():
            raise ProjectError(
                f"Project '{project_name}' does not exist."
            )
        try:
            shutil.rmtree(project_path)
            log.info(
                "project_deleted",
                project=project_name,
            )
        except OSError as e:
            raise ProjectError(
                f"Failed to delete project '{project_name}': {e}"
            ) from e


    def project_exists(self, project_name: str) -> bool:
        project_name = self._validate_name(project_name)

        return (PROJECTS_DIR / project_name).is_dir()

    def list_projects(self) -> list[str]:

        if not PROJECTS_DIR.exists():
            return []

        try:
            return sorted(
                p.name
                for p in PROJECTS_DIR.iterdir()
                if p.is_dir()
            )

        except OSError as e:
            raise ProjectError(
                f"Failed to list projects: {e}"
            ) from e


    def get_project_path(self, project_name: str) -> Path:
   
        project_name = self._validate_name(project_name)

        project_path = PROJECTS_DIR / project_name

        if not project_path.exists():
            raise ProjectError(
                f"Project '{project_name}' does not exist."
            )

        return project_path


    def get_workspace(self, project_name: str) -> dict[str, Path]:

        root = self.get_project_path(project_name)

        return {
            "root": root,
            "cache": root / CACHE_DIR,
            "vectors": root / VECTORS_DIR,
            "templates": root / TEMPLATES_DIR,
            "outputs": root / OUTPUTS_DIR,
        }


    @staticmethod
    def _validate_name(project_name: str) -> str:

        if not project_name:
            raise ProjectError(
                "Project name cannot be empty."
            )

        valid = all(
            c.isalnum() or c in {"_", "-"}
            for c in project_name
        )

        if not valid:
            raise ProjectError(
                f"Invalid project name: '{project_name}'."
            )

        return project_name