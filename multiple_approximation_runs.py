from build_DS import find_governing_equations
from get_vector_field import get_equation
import numpy as np
import pandas as pd

dim = 2
bounds = [1.6, 4]
# bounds = [0.5, 0.5, 0.5]
# n_x, n_y
tol = 1e-1
# act = 'elu'
eq = 'van_der_pool_2d'
vf = get_equation(eq)
# time series hyperparameters
dt = 0.1
# init= [0, 1]
t_end = 20
freq = 25
deg = 100
lamda = 0.1
n_traj = 20
verbose = True

runs = {'n': [], 'traj': [], 'mse_sindy_x1': [], 'mse_sindy_x2': [],'mse_mlp_x1': [],'mse_mlp_x2': [], 'time_S': [], 'time_M': []}
for dt in  [0.01, 0.02, 0.04, 0.05, 0.1, 0.5]:
# for dt in  [0.5]:
    for traj in [10, 20, 50, 100, 500]:
    # for traj in [10]:
        n_points = np.arange(0, t_end, dt).shape[0]
        print('---- {} ----- {}'.format(n_points, traj))
        GE = find_governing_equations(func = get_equation('van_der_pool_1d'), bounds = bounds,
                                        dt= dt, t_end=t_end, dim = dim, n_traj = traj,  path = '.', verbose = False)
        GE.create_time_series(multiple = True)
        model, end_time = GE.find_equations(freq, deg, lamda, plot = False)
        x, y = GE.create_time_series_MLP(dim)
        train_dataset, validation_dataset = GE.process_MLP_inputs(x,y)
        MLP_model, mlp_end_time = GE.find_equations_MLP(train_dataset, validation_dataset)
        res = GE.evaluate_models(vf, [50,50])
        runs['n'].append(n_points)
        runs['traj'].append(traj)
        runs['mse_sindy_x1'].append(res[0])
        runs['mse_sindy_x2'].append(res[1])
        runs['mse_mlp_x1'].append(res[2])
        runs['mse_mlp_x2'].append(res[3])
        runs['time_S'].append(end_time)
        runs['time_M'].append(mlp_end_time)

pd.DataFrame.from_dict(data=runs).to_csv(
            'approx_results.csv', index = True, header=runs.keys())