import importlib.util

for name in ("triangle", "shapely", "mapbox_earcut", "meshpy", "trianglelib"):
    print(name, bool(importlib.util.find_spec(name)))
