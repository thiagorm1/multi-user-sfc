from enum import Enum

SHAREABLE_PREFIXES: tuple[str, ...] = ("IA_DET_FT_", "RE_region_", "MA_region_")
# =========================================================================
# ENUMS DE SEGURANÇA (Refatoração: Eliminação de Magic Strings)
# =========================================================================

class NodeType(str, Enum):
    """Contrato estrito para as categorias de nós na infraestrutura."""
    SERVER = "server"
    ROUTER = "router"
    MOBILE_DEVICE = "mobile_device"

class NodeLevel(str, Enum):
    """Contrato estrito para a categoria de confiabilidade dos nós físicos."""
    A = "a"
    B = "b"
    C = "c"
    DEFAULT = "default"
