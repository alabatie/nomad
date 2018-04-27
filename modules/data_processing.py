import pandas as pd
import numpy as np

import itertools

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def add_feat(array, feat):
    """
        Add features to an existing feature vector
    """
    return np.hstack((array, np.array(feat).reshape((3000, -1))))


def make_Xy():
    """
        Create input and output data from csv files
    """
    filepath = "data/train.csv"
    df_train = pd.read_csv(filepath, index_col=0)

    filepath = "data/test.csv"
    df_test = pd.read_csv(filepath, index_col=0)

    df = pd.concat((df_train, df_test))

    # one-hot encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df['spacegroup'])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # construct X
    X = np.zeros((3000, 0))

    X = add_feat(X, onehot_encoded)

    for col_name in ['percent_atom_in', 'percent_atom_ga', 'percent_atom_al']:
        X = add_feat(X, df[col_name]*df['number_of_total_atoms'])

    for col_name in ['percent_atom_in', 'percent_atom_ga', 'percent_atom_al', 'number_of_total_atoms']:
        X = add_feat(X, df[col_name])

    for col_name in ['lattice_vector_3_ang', 'lattice_vector_2_ang', 'lattice_vector_1_ang',
                     'lattice_angle_gamma_degree', 'lattice_angle_beta_degree', 'lattice_angle_alpha_degree']:
        X = add_feat(X, df[col_name])

    for col_name in ['lattice_angle_gamma_degree', 'lattice_angle_beta_degree', 'lattice_angle_alpha_degree']:
        X = add_feat(X, np.cos(np.pi/180.0*df[col_name]))
    for col_name in ['lattice_angle_gamma_degree', 'lattice_angle_beta_degree', 'lattice_angle_alpha_degree']:
        X = add_feat(X, np.sin(np.pi/180.0*df[col_name]))

    # normalize X to be in a compatible range for the net at initilization
    X /= X.max(axis=0, keepdims=True)

    # construct y
    y = np.zeros((3000, 0))
    y = add_feat(y, df['formation_energy_ev_natom'])
    y = add_feat(y, df['bandgap_energy_ev'])

    Xsub = X[np.isnan(y[:, -1]), ]  # submission samples = samples for which we don't know y values
    id_sub = df.index[np.isnan(y[:, -1])]

    X = X[~np.isnan(y[:, -1]), ]  # training samples = samples for which we know y values
    y = y[~np.isnan(y[:, -1]), ]

    return X, y, Xsub, id_sub


def post_process(pred):
    """
        Postprocessing = undo log transform + threshold at 0
    """
    pred = pred * (pred > 0)
    pred = np.exp(pred) - 1.0
    return pred


def RMSLE(y, pred):
    """
        Final evaluation metric
    """
    rmsle = []
    for icol in range(y.shape[1]):
        diff = (np.log(1+pred[:, icol]) - np.log(1+y[:, icol]))**2
        ind = ~np.isnan(diff)
        rmsle.append(np.sqrt(diff[ind].mean()))

    return rmsle


def make_DA(X, y=None, ind=None, sub=False):
    """
        Generate random data-augmented versions of inputs
    """
    # load cube
    cube_DA = np.load("data/cube.npy")
    cube_DA = (cube_DA - cube_DA.mean()) / cube_DA.std()  # normalize

    if sub:
        # if submission time, we select test samples and y is simply nan
        cube_DA = cube_DA[-600:, ]
        X_DA = np.array(X)
        y_DA = np.tile(np.nan, (600, 2))
    else:
        # otherwise we use input y and select only ind
        cube_DA = cube_DA[:2400, ][ind, ]
        X_DA = np.array(X)
        y_DA = np.array(y)

    # get sizes
    N = X_DA.shape[0]
    nb = cube_DA.shape[1]

    # random rolling and flipping in all directions
    roll_sample = np.random.randint(0, nb, size=(N, 3))
    flip_sample = np.random.random((N, 3)) > 0.5
    ind_nb = range(nb)
    for isample in range(N):
        rollx = np.roll(ind_nb, roll_sample[isample, 0])
        rolly = np.roll(ind_nb, roll_sample[isample, 1])
        rollz = np.roll(ind_nb, roll_sample[isample, 2])

        cube_DA[isample, ] = cube_DA[isample, rollx, :, :, :]
        cube_DA[isample, ] = cube_DA[isample, :, rolly, :, :]
        cube_DA[isample, ] = cube_DA[isample, :, :, rollz, :]

        cube_DA[isample, ] = cube_DA[isample, ::-1, :, :, :] if flip_sample[isample, 0] else cube_DA[isample, ]
        cube_DA[isample, ] = cube_DA[isample, :, ::-1, :, :] if flip_sample[isample, 1] else cube_DA[isample, ]
        cube_DA[isample, ] = cube_DA[isample, :, :, ::-1, :] if flip_sample[isample, 2] else cube_DA[isample, ]

    # random permutation of the axes
    p_list = list(itertools.permutations([0, 1, 2]))

    X_DA = np.tile(X_DA, (6, 1))
    y_DA = np.tile(y_DA, (6, 1))
    cube_DA = np.tile(cube_DA, (6, 1, 1, 1, 1))
    for ip, p in enumerate(p_list):
        for isample in range(N):
            p = list(p)
            cube_DA[ip*N+isample, ] = np.transpose(cube_DA[ip*N+isample, ], (p[0], p[1], p[2], 3))

            X_DA[ip*N+isample, -12:-9] = X_DA[ip*N+isample, -12:-9][p]
            X_DA[ip*N+isample, -9:-6] = X_DA[ip*N+isample, -9:-6][p]
            X_DA[ip*N+isample, -6:-3] = X_DA[ip*N+isample, -6:-3][p]
            X_DA[ip*N+isample, -3:] = X_DA[ip*N+isample, -3:][p]

    return X_DA, y_DA, cube_DA
