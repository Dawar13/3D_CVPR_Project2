"""rsgen: synthetic rolling-shutter frames rendered from 3D meshes (EXP2 §6).

core    camera, shutter timing, sampled trajectories, the ray-casting renderer
scenes  mesh loading (Replica-style vertex-coloured PLY), a procedural test room, free-space queries
motion  trajectory generator (handheld, drone, pure rotation, pure translation, mixed) with speed targets
postfx  noise / gamma, crop / resize with row-timing bookkeeping, simulated electronic stabilisation
"""
