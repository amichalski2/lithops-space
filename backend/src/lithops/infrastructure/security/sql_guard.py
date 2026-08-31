"""Conservative SQL allow-list used by CEO-Bench read-only queries."""

import re

from lithops.domain.errors import BenchmarkContractError

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|grant|revoke|truncate|copy|call|do|merge)\b",
    re.IGNORECASE,
)


def validate_readonly_sql(sql: str) -> str:
    statement = sql.strip()
    if not statement:
        raise BenchmarkContractError("query must not be empty")

    normalized = statement[:-1].rstrip() if statement.endswith(";") else statement
    if ";" in normalized:
        raise BenchmarkContractError("multiple SQL statements are not allowed")
    if not re.match(r"^(select|with)\b", normalized, flags=re.IGNORECASE):
        raise BenchmarkContractError("only SELECT or read-only WITH queries are allowed")
    if _WRITE_KEYWORDS.search(normalized):
        raise BenchmarkContractError("query contains a write or DDL keyword")
    return normalized
