"""Shared construction yard geometry, in metres with Z up."""

ZONES = {
    "A": {"name": "Materials", "position": [-5.0, 4.0, 4.5]},
    "B": {"name": "Fuel depot", "position": [5.0, 4.0, 4.5]},
    "C": {"name": "Equipment", "position": [5.0, -4.0, 4.5]},
    "D": {"name": "Excavation", "position": [-5.0, -4.0, 4.5]},
}
HOME = [0.0, -6.3, 0.12]
BOUNDS = {"x": [-9.0, 9.0], "y": [-8.0, 8.0], "z": [0.0, 7.0]}
OBSTACLES = [
    {"name": "site office", "position": [-7.0, 0.0, 1.1], "size": [1.2, 2.0, 1.1], "color": "#d7ddd9"},
    {"name": "shipping container", "position": [7.0, 0.0, 1.0], "size": [1.0, 2.0, 1.0], "color": "#c57b3b"},
    {"name": "material stack", "position": [-6.0, 5.4, 0.4], "size": [1.4, 0.5, 0.4], "color": "#bda17c"},
    {"name": "concrete pipes", "position": [-3.7, 5.4, 0.45], "size": [0.6, 0.65, 0.45], "color": "#969e9c"},
    {"name": "fuel tank", "position": [6.0, 5.5, 0.65], "size": [1.0, 0.5, 0.65], "color": "#c2c9c7"},
    {"name": "excavator", "position": [6.4, -5.0, 0.7], "size": [0.65, 1.0, 0.7], "color": "#e8b33e"},
    {"name": "pallets", "position": [3.8, -5.5, 0.35], "size": [0.6, 0.7, 0.35], "color": "#bda17c"},
]

def description():
    return {"zones": ZONES, "home": HOME, "bounds": BOUNDS,
            "obstacles": OBSTACLES, "hazards": {},
            "physics": {"engine": "MuJoCo", "physicsHz": 500, "controlHz": 250}}
