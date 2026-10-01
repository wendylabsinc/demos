"""Convert the attributed CesiumMan GLB to one textured MuJoCo visual mesh.

The skeleton's bind pose is applied to the source vertices. The output is a
static standing visual; locomotion belongs to the simulator's physical body.
Run once with NumPy and Pillow. Neither conversion code nor GLB loads at runtime.
"""
import io
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "assets" / "person"
raw = (ROOT / "CesiumMan.glb").read_bytes()
assert struct.unpack_from("<I", raw, 0)[0] == 0x46546C67
chunks, offset = {}, 12
while offset < len(raw):
    length, kind = struct.unpack_from("<II", raw, offset)
    chunks[kind] = raw[offset+8:offset+8+length]
    offset += 8+length
gltf = json.loads(chunks[0x4E4F534A])
binary = chunks[0x004E4942]


def accessor(index):
    item = gltf["accessors"][index]
    view = gltf["bufferViews"][item["bufferView"]]
    kinds = {5120:"i1", 5121:"u1", 5122:"<i2", 5123:"<u2", 5125:"<u4", 5126:"<f4"}
    columns = {"SCALAR":1, "VEC2":2, "VEC3":3, "VEC4":4, "MAT4":16}[item["type"]]
    dtype = np.dtype(kinds[item["componentType"]])
    stride = view.get("byteStride", columns*dtype.itemsize)
    result = np.ndarray((item["count"], columns), dtype=dtype, buffer=binary,
        offset=view.get("byteOffset", 0)+item.get("byteOffset", 0),
        strides=(stride, dtype.itemsize)).copy()
    return result[:, 0] if columns == 1 else result


def matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"]).reshape(4,4).T
    x,y,z,w = node.get("rotation", [0,0,0,1])
    result = np.eye(4)
    result[:3,:3] = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    result[:3,:3] *= np.array(node.get("scale", [1,1,1]))
    result[:3,3] = node.get("translation", [0,0,0])
    return result


# Bake a supplied walk-animation frame so arms do not remain in the bind T-pose.
for channel in gltf["animations"][0]["channels"]:
    sampler = gltf["animations"][0]["samplers"][channel["sampler"]]
    times, values = accessor(sampler["input"]), accessor(sampler["output"])
    target_time = min(.35, float(times[-1]))
    i = max(0, min(len(times)-2, int(np.searchsorted(times,target_time)-1)))
    blend = (target_time-times[i])/(times[i+1]-times[i])
    a,b = values[i].astype(float), values[i+1].astype(float)
    path = channel["target"]["path"]
    if path == "rotation" and np.dot(a,b) < 0:
        b = -b
    value = a+(b-a)*blend
    if path == "rotation":
        value /= np.linalg.norm(value)
    gltf["nodes"][channel["target"]["node"]][path] = value.tolist()

world = {}
def visit(index, parent):
    node = gltf["nodes"][index]
    world[index] = parent@matrix(node)
    for child in node.get("children", []):
        visit(child, world[index])
for node in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
    visit(node, np.eye(4))
node_index = next(i for i,node in enumerate(gltf["nodes"]) if "mesh" in node)
node = gltf["nodes"][node_index]
primitive = gltf["meshes"][node["mesh"]]["primitives"][0]
attributes = primitive["attributes"]
vertices = accessor(attributes["POSITION"])
uv = accessor(attributes["TEXCOORD_0"])
faces = accessor(primitive["indices"]).reshape(-1,3)
skin = gltf["skins"][node["skin"]]
inverse = accessor(skin["inverseBindMatrices"]).reshape(-1,4,4).transpose(0,2,1)
joints, weights = accessor(attributes["JOINTS_0"]), accessor(attributes["WEIGHTS_0"])
local_inverse = np.linalg.inv(world[node_index])
joint_matrices = np.array([local_inverse@world[j]@inverse[i] for i,j in enumerate(skin["joints"])])
homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
skinned = np.zeros_like(homogeneous)
for k in range(4):
    skinned += np.einsum('nij,nj->ni', joint_matrices[joints[:,k]], homogeneous)*weights[:,k,None]
vertices = skinned[:,:3]
# CesiumMan's mesh coordinates are Z-up and face positive X. Normalize height.
vertices[:,2] -= vertices[:,2].min()
vertices *= 1.7 / vertices[:,2].max()
vertices[:,:2] -= (vertices[:,:2].max(axis=0)+vertices[:,:2].min(axis=0))/2
with (ROOT / "person.obj").open("w") as output:
    output.write("# CesiumMan, Copyright 2017 Cesium. CC-BY-4.0. Skinned/normalized for MuJoCo.\n")
    for vertex in vertices:
        output.write("v %.7f %.7f %.7f\n" % tuple(vertex))
    for u,v in uv:
        output.write("vt %.7f %.7f\n" % (u,1-v))
    for face in faces+1:
        output.write("f %d/%d %d/%d %d/%d\n" % tuple(value for i in face for value in (i,i)))
texture_index = gltf["materials"][primitive.get("material",0)]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
image = gltf["images"][gltf["textures"][texture_index]["source"]]
view = gltf["bufferViews"][image["bufferView"]]
image_bytes = binary[view.get("byteOffset",0):view.get("byteOffset",0)+view["byteLength"]]
Image.open(io.BytesIO(image_bytes)).convert("RGB").save(ROOT / "person.png")
print("Converted", len(vertices), "vertices; bounds", vertices.min(axis=0), vertices.max(axis=0))
