import numpy as np

nb = 12  # number of bins in each cube axis
sigma = 0.05  # size of kernel around each atom
b = (np.arange(nb) + 0.5) / float(nb)  # binning vector
atom_list = ['Ga', 'Al', 'In', 'O']
integral = None

# array of possible shifts to take into account lattice periodicity into 3D cube
shift_array = [np.array([sx, sy, sz]) for sx in [-1, 0, 1] for sy in [-1, 0, 1] for sz in [-1, 0, 1]]
shift_array = np.array(shift_array)


def compute_integral():
    """
        This function computes the integral of the kernel function around an atom
            - we start by subpixeling a cube pixel
            - we also discretize distance in (x, y, z) with a grid ranging from 0 to 5*sigma
            - for every possible distance on the grid (dx, dy, dz),
              we compute the average value of the kernel on all subpixels
            - we approximate the integral on the cube pixel with this average value
    """
    global integral

    dxyz = np.arange(-5, 6) / 11. / float(nb)

    sub_cubex = np.tile(dxyz.reshape((11, 1, 1)), (1, 11, 11))  # x coordinate of subpixel
    sub_cubey = np.tile(dxyz.reshape((1, 11, 1)), (11, 1, 11))  # y coordinate of subpixel
    sub_cubez = np.tile(dxyz.reshape((1, 1, 11)), (11, 11, 1))  # z coordinate of subpixel

    d = np.linspace(0, 5*sigma, 200)  # grid of distance values
    integral = np.tile(np.nan, (200, 200, 200))  # will store integral values for all possible distances

    for ix, x in enumerate(d):
        for iy, y in enumerate(d):
            for iz, z in enumerate(d):
                if ix > iy or iy > iz:
                    continue
                distx = x - sub_cubex
                disty = y - sub_cubey
                distz = z - sub_cubez
                f = np.exp(-0.5*(distx**2+disty**2+distz**2)/sigma**2)  # value of kernel function
                f_mean = f.mean()  # average

                # use symmetry to set all values corresponding to this combination of (dx, dy, dz)
                integral[ix, iy, iz] = f_mean
                integral[ix, iz, iy] = f_mean
                integral[iy, ix, iz] = f_mean
                integral[iy, iz, ix] = f_mean
                integral[iz, ix, iy] = f_mean
                integral[iz, iy, ix] = f_mean


def get_pos_lattice(filename):
    """
        Get position of all lattice atoms
         - expressed in the basis of lattice vectors
         - all atom coordinates are thus in the range [0, 1]
    """
    pos = {atom: [] for atom in atom_list}
    lattice_vec = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                atom = x[4]
                pos[atom].append(np.array(x[1:4], dtype=np.float))
            elif x[0] == 'lattice_vector':
                lattice_vec.append(np.array(x[1:4], dtype=np.float))

    lattice_matrix = np.zeros((3, 3))
    lattice_matrix[:, 0], lattice_matrix[:, 1], lattice_matrix[:, 2] = lattice_vec[0], lattice_vec[1], lattice_vec[2]

    pos_lattice = {atom: [] for atom in atom_list}
    for atom, pos_atom in pos.items():
        for posi in pos_atom:
            pos_lattice[atom].append(np.linalg.solve(lattice_matrix, posi))

    return pos_lattice


def periodic_pos(pos_list):
    """
        For every position in list, add positions shifted with lattice periodicity
            - only keep shifted position that are not further than 5*sigma from each border
    """
    pos_periodic_list = []
    for pos in pos_list:
        pos_array = shift_array + pos

        # only keep shifted positions not further than 5*sigma from each border
        ind = (pos_array < -5*sigma) + (pos_array > 1+5*sigma)
        ind = ind.sum(1) == 0
        ind = np.where(ind)[0]

        for i in ind:
            pos_periodic_list.append(pos_array[i, :])
    return pos_periodic_list


def make_cube_pos(pos):
    """
        Get cube generated from a single position
            - First get distance from position to each pixel
            - Then convert to integral distance indices for each pixel
            - Finally convert to integral value for each pixel
    """
    x, y, z = pos[0], pos[1], pos[2]

    # cube generated from this position alone
    cube_pos = np.tile(np.nan, (nb, nb, nb))

    # get distance from position at each pixel of the cube
    dx, dy, dz = np.abs(x-b), np.abs(y-b), np.abs(z-b)
    dx = np.tile(dx.reshape(nb, 1, 1), (1, nb, nb))
    dy = np.tile(dy.reshape(1, nb, 1), (nb, 1, nb))
    dz = np.tile(dz.reshape(1, 1, nb), (nb, nb, 1))

    # get corresponding indices of distance for each pixel
    idx = np.round(dx / (5*sigma) * 200).astype(int)
    idy = np.round(dy / (5*sigma) * 200).astype(int)
    idz = np.round(dz / (5*sigma) * 200).astype(int)

    idx = np.minimum(idx, 199)
    idy = np.minimum(idy, 199)
    idz = np.minimum(idz, 199)

    # assign the value of the integral for each pixel, depending on its indices of distance
    cube_pos = integral[idx, idy, idz]

    return cube_pos


def make_cube(filename):
    """
        Function for creating cube from filename containing all atom positions
            - First get atom positions from file
            - Append shifted positions with periodic conditions
            - Finally generate cube with all these positions
            - In the final cube, each atom type corresponds to a channel
    """
    pos_lat_atom = get_pos_lattice(filename)

    cube_atom = {atom: np.zeros((nb, nb, nb)) for atom in atom_list}
    for atom, pos_list in pos_lat_atom.items():
        pos_list = periodic_pos(pos_list)  # append shifted positions with periodic conditions
        for pos in pos_list:
            cube_atom[atom] += make_cube_pos(pos)

    # in the final cube, each atom type corresponds to a channel
    cube = np.zeros((nb, nb, nb, 4))
    for iatom, atom in enumerate(atom_list):
        cube[:, :, :, iatom] = cube_atom[atom]

    return cube
