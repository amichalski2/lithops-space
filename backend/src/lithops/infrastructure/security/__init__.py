"""Safety boundaries for external tools and untrusted inputs."""

from lithops.infrastructure.security.sql_guard import validate_readonly_sql

__all__ = ["validate_readonly_sql"]
