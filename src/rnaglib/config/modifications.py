from importlib import resources
from . import cache

def get_modifications_cache():
    with resources.open_text(cache, 'modifications_cache.json') as file:
        modifications = file.read()

    return eval(modifications)