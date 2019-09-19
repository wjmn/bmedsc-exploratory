from modules.grid import CartesianGrid, PolarGrid

GRIDS = [
    # 3 sets of random half polar grids
    (PolarGrid, 8, 8, 48, 48, True, True),
    (PolarGrid, 8, 8, 48, 48, True, True),
    (PolarGrid, 8, 8, 48, 48, True, True),

    # 1 set of non-random half polar grids
    (PolarGrid, 8, 8, 48, 48, True, False),

    # 1 set of random cartesian grids
    (CartesianGrid, 8, 8, 48, 48, True, True),

    # 1 set of non-random cartesian grids
    (CartesianGrid, 8, 8, 48, 48, True, False),
]

for grid_init, *args in GRIDS:
    grid = grid_init(*args)
    print(grid.grid_id)
    grid.save('../grids/')
