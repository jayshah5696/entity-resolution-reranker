COLVAL_FIELD_MAP = {
    "fn": "first_name",
    "ln": "last_name",
    "org": "company",
    "title": "title",
    "country": "country"
}

COLVAL_ORDER = ["fn", "ln", "org", "title", "country"]

PIPE_ORDER = ["first_name", "last_name", "company", "email", "country"]

def colval_serialize(record: dict) -> str:
    """Serializes a record to COL VAL format."""
    parts = []
    for short_col in COLVAL_ORDER:
        full_col = COLVAL_FIELD_MAP[short_col]
        val = record.get(full_col)
        
        if val is None:
            continue
            
        if isinstance(val, str) and not val.strip():
            continue
            
        parts.append(f"COL {short_col} VAL {val}")
        
    return " ".join(parts)

def colval_pair(record_a: dict, record_b: dict) -> str:
    """Joins two serialized records with [SEP]."""
    return f"{colval_serialize(record_a)} [SEP] {colval_serialize(record_b)}"

def pipe_serialize(record: dict) -> str:
    """Phase 1 compatibility serialization."""
    parts = []
    for field in PIPE_ORDER:
        parts.append(str(record.get(field, "")))
    return " | ".join(parts)
