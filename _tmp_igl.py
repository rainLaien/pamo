import igl

print(hasattr(igl, "lscm"))
if hasattr(igl, "lscm"):
    print(igl.lscm.__doc__)
print("triangle", hasattr(igl, "triangle"))
print("triangulate", hasattr(igl, "triangulate"))
