"""
This program runs the model from end-to-end. Toggle any parameters you want 
to change from this program. Default parameters run the 2-d Van der Pol equation.
"""
from build_DS import find_governing_equations
from get_vector_field import get_equation
from NN_Lyapunov import build_lyapunov, LyapunovModel
import tensorflow as tf
import numpy as np
import os
import subprocess
import pandas as pd
np.random.seed(222)

if __name__ == "__main__":

    try: 
        if not os.path.exists("Experiments"):
            os.makedirs("Experiments")

        # choose equation to evaluate here
        eq = 'van_der_pool_2d'
        # toggle for trainable weights of the first layer of the network here
        train = False
        # DyS: choose between 'True', 'SINDy', 'MLP' for the model to have access to
        # the true evolution equations, SINDy method and MLP method respectively.
        DyS = 'True'
        path = 'Experiments/Lyapunov_eq_{}_train_{}_DyS_{}'.format(eq, train, DyS)

        def uniquify(path):
            # creates unique folder for results
            counter = 1
            while os.path.exists(path):
                path = path.split()[0] + " (" + str(counter) + ")"
                counter += 1

            return path
        # create new file path if it doesn't exist
        path = uniquify(path)

        if not os.path.exists(path):
        
            # if the demo_folder directory is not present 
            # then create it.
            os.makedirs(path)
        print('path is: {}'.format(path))
        
        "Model Hyperparameters"
        hidden_u = []
        # model width
        m = 200
        # model dimensions
        dim = 2
        bounds = [1.6, 4]
        # bounds = np.ones(dim)
        # number of bounds for each dimension
        n = [6, 15]
        n_t = [10, 10]
        # n = np.ones(dim, dtype=int) *3
        # n_t = n
        batch_n = np.prod(n)
        buff = None
        epochs = 10000
        tol = 1e-1
        # activation function to use
        act = tf.math.cos
        if not train:
            lr = m * 0.1
            opt = tf.keras.optimizers.SGD(learning_rate=lr, nesterov=True)
        else:
            lr = 0.01
            opt = tf.keras.optimizers.Adam(lr)
        vf = get_equation(eq)
        # time series hyperparameters used when DyS is either SINDy or MLP
        dt = 0.02
        # until what time to simulate data
        t_end = 20
        # number of trajectories
        traj = 5
        #SINDy search space
        freq = 25
        deg = 25
        lamda = 0.1

        params = {'eq': eq, 'm': m, 'dim': dim, 'bounds0': bounds[0], 'bounds1': bounds[1], 
                'n_x': n[0], 'n_y': n[1], 'n_x_t': n_t[0], 'n_y_t': n_t[1], 'batch_n': batch_n, 
                'epochs': epochs, 'tol': tol, 'activation': str(act), 'train_params': train, 
                'opt':  str(opt.get_slot_names)[:-27][-3:],  'lr': lr, 'DyS': DyS,
                'dt': dt, 't_end': t_end, 'freq': freq, 'deg': deg, 'lamda': lamda}


        with open(path + "/model_params.txt", 'w') as f: 
            for key, value in params.items(): 
                f.write('%s:%s\n' % (key, value))

        if DyS == 'SINDy':
            GE = find_governing_equations(func = get_equation('van_der_pool_1d'), bounds = bounds,
                                            dt= dt, t_end=t_end, dim = dim, n_traj = traj,  path = path, verbose = True)
            GE.create_time_series(multiple = True)
            model, end_time = GE.find_equations(freq, deg, lamda, plot = False)
            # evaluate now
            mse_x, mse_a = GE.evaluate_models(vf, n_t, plot_results= True)
            runs = {'DyS': DyS, 'n_t': n_t, 'dt': dt, 'traj': traj, 'mse_x': mse_x, 'mse': mse_a}
            
            pd.DataFrame.from_dict(data=runs).to_csv(
            path + '/approx_results.csv', index = True, header=runs.keys())

            def vf(x):
                if type(x) != np.ndarray:
                    x = x.numpy()
                return model.predict(x).T
        
        if DyS == 'MLP':
            print('IN MLP...')
            GE = find_governing_equations(func = get_equation('van_der_pool_1d'), bounds = bounds,
                                            dt= dt, t_end=t_end, dim = dim, n_traj = traj,  path = path, verbose = True)
            x, y = GE.create_time_series_MLP(dim)
            train_dataset, validation_dataset = GE.process_MLP_inputs(x,y)
            MLP_model, mlp_end_time = GE.find_equations_MLP(train_dataset, validation_dataset)
            mse_x, mse_a = GE.evaluate_models(vf, n_t, plot_results= True)
            runs = {'DyS': DyS, 'n_t': n_t, 'dt': dt, 'traj': traj, 'mse_x': mse_x, 'mse': mse_a}
            
            pd.DataFrame.from_dict(data=runs).to_csv(
            path + '/approx_results.csv', index = True, header=runs.keys())
            
            def vf(x):
                if type(x) != np.ndarray:
                    x = x.numpy()
                return MLP_model.predict(x).T

        base_model = build_lyapunov(vf, dim, act, bounds, m, epochs, tol, path, func_name = eq)

        train_dataset, train_data_points, train_input_RHS = base_model.create_dataset(n, 
                                        dim, bounds, batch_n, buff = None, train = True, plot=False)
        test_dataset, test_data_points, test_input_RHS = base_model.create_dataset(n_t, 
                                        dim, bounds, batch_n, buff = None, train = False, plot=False)

        my_mlp = base_model.get_regularised_bn_mlp(hidden_u, LyapunovModel, opt= opt,  train=train)
        all_loss_values, all_test_loss_values, fitted_model = base_model.fit(train_dataset, test_dataset, plot=True)
        fitted_model.save(path + '/Lyapunov_{}d_{}m_{}epochs_opt_{}_lr_{}_eq_{}.h5'.format(dim, 
                                m,epochs, str(opt.get_slot_names)[:-27][-3:], lr, eq))

        # use this for 2-d
        mse = base_model.plot_solution(20)
        base_model.plot_Layer(20)
        # use this for any dimension
        mse = base_model.plot_solution_ND(n_t)
        # use this for dim>3
        # base_model.psnd(30, 1,7)

        results = {'all_loss_values': all_loss_values, 
                'all_test_loss_values': all_test_loss_values,
                'activation': str(act),
                'm': m,
                'n': batch_n,
                'mse': mse}
        
        pd.DataFrame.from_dict(data=results).to_csv(
            path +'/results_m_{}_act_{}_n_{}.csv'.format(m, str(act), str(batch_n)), index = True, header=results.keys())

    except KeyboardInterrupt:
        # remove results directory if the program is exited
        import shutil
        shutil.rmtree(path)
