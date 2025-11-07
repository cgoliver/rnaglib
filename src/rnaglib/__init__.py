"""RNA Geometric Library - Tools for learning on RNA 3D structures."""

__version__ = "3.4.10"

# Import submodules to make them available for autodoc
try:
    from . import dataset
except ImportError:
    pass

try:
    from . import tasks
except ImportError:
    pass

try:
    from . import transforms
except ImportError:
    pass

try:
    from . import dataset_transforms
except ImportError:
    pass

try:
    from . import algorithms
except ImportError:
    pass

try:
    from . import prepare_data
except ImportError:
    pass

try:
    from . import utils
except ImportError:
    pass

try:
    from . import learning
except (ImportError, OSError):
    # OSError can occur with torch_scatter compatibility issues
    pass

