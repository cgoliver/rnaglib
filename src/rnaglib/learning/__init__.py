from .task_models import PygModel
try:
    from .gvp import GVPModel
except (ImportError, OSError):
    # OSError can occur with torch_scatter compatibility issues
    # ImportError can occur if torch_scatter is not available
    GVPModel = None
