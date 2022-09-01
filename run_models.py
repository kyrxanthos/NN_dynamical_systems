from pickle import FALSE
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

        eq = 'van_der_pool_2d'
        # eq = 'simple_2d'
        # eq = 'kevin_3d'
        # eq = 'cpa_3d'
        eq = 'complicated_10d'
        train = False
        # use_true_equation = False
        DyS = 'True'
        # DyS = 'SINDy'
        # DyS = 'MLP'
        path = 'Experiments2/Lyapunov_eq_{}_train_{}_DyS_{}'.format(eq, train, DyS)

        def uniquify(path):
            # filename, extension = os.path.splitext(path)
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
        m = 3200
        dim = 10
        # bounds = [1.6, 4]
        # bounds = [0.5, 0.5, 0.5]
        bounds = np.ones(dim)
        # n_x, n_y
        n = [6, 15]
        n_t = [10, 10]
        # n = [5, 5, 5]
        n = np.ones(dim, dtype=int) *3
        n_t = n
        batch_n = 1024
        buff = None
        epochs = 100
        tol = 1e-1
        act = tf.math.cos
        # act = 'elu'
        if not train:
            lr = m * 0.1
            opt = tf.keras.optimizers.SGD(learning_rate=lr, nesterov=True)
        else:
            lr = 0.01
            opt = tf.keras.optimizers.Adam(lr)
        vf = get_equation(eq)
        # time series hyperparameters
        dt = 0.02
        # init= [0, 2]
        t_end = 20
        freq = 25
        deg = 100
        lamda = 0.1
        traj = 500

        params = {'eq': eq, 'm': m, 'dim': dim, 'bounds0': bounds[0], 'bounds1': bounds[1], 
                'n_x': n[0], 'n_y': n[1], 'n_x_t': n_t[0], 'n_y_t': n_t[1], 'batch_n': batch_n, 
                'epochs': epochs, 'tol': tol, 'activation': str(act), 'train_params': train, 
                'opt':  str(opt.get_slot_names)[:-27][-3:],  'lr': lr, 'DyS': DyS,
                'dt': dt, 't_end': t_end, 'freq': freq, 'deg': deg, 'lamda': lamda}


        with open(path + "/model_params.txt", 'w') as f: 
            for key, value in params.items(): 
                f.write('%s:%s\n' % (key, value))

        # insert kernel here aswell

        # if not use_true_equation:
        #     GE = find_governing_equations(func = get_equation('van_der_pool_1d'), bounds = bounds,
        #                                          dt= dt, t_end=t_end, dim = dim, path = path)
        #     GE.create_time_series(init = init)
        #     model = GE.find_equations(freq, deg, lamda, verbose = True, plot = True)

        #     def vf(x):
        #         if type(x) != np.ndarray:
        #             x = x.numpy()
        #         return model.predict(x).T

        if DyS == 'SINDy':
            GE = find_governing_equations(func = get_equation('complicated_10d_simple'), bounds = bounds,
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
                    # print(x.shape)
                    # assert not np.isnan(x).any()
                    y = model.predict(x).T
                    # assert not np.isnan(y).any()
                    # print(y.shape)
                return model.predict(x).T
        
        if DyS == 'MLP':
            print('IN MLP...')
            GE = find_governing_equations(func = get_equation('complicated_10d_simple'), bounds = bounds,
                                            dt= dt, t_end=t_end, dim = dim, n_traj = traj,  path = path, verbose = True)
            x, y = GE.create_time_series_MLP(dim)
            train_dataset, validation_dataset = GE.process_MLP_inputs(x,y)
            MLP_model, mlp_end_time = GE.find_equations_MLP(train_dataset, validation_dataset)
            # evaluate now
            mse_x, mse_a = GE.evaluate_models(vf, n_t, plot_results= True)
            runs = {'DyS': DyS, 'n_t': n_t, 'dt': dt, 'traj': traj, 'mse_x': mse_x, 'mse': mse_a}
            
            pd.DataFrame.from_dict(data=runs).to_csv(
            path + '/approx_results.csv', index = True, header=runs.keys())
            
            # add here evaluation of the approximation with just the mses (plot =False)
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
        # print(my_mlp.summary())
        # base_model.plot_Layer(20)
        all_loss_values, all_test_loss_values, fitted_model = base_model.fit(train_dataset, test_dataset, plot=True)
        fitted_model.save(path + '/Lyapunov_{}d_{}m_{}epochs_opt_{}_lr_{}_eq_{}.h5'.format(dim, m,epochs, str(opt.get_slot_names)[:-27][-3:], lr, eq))
        # mse = base_model.plot_solution(20)
        # mse = base_model.plot_solution3_D(25)
        mse = base_model.plot_solution_ND(n_t)
        base_model.psnd(30, 1,7)
        base_model.psnd(30, 2,5)
        base_model.psnd(30, 4,9)
        base_model.psnd(30, 3,8)
        # base_model.plot_Layer(20)

        results = {'all_loss_values': all_loss_values, 
                'all_test_loss_values': all_test_loss_values,
                'activation': str(act),
                'm': m,
                'n': batch_n,
                'mse': mse}
        
        pd.DataFrame.from_dict(data=results).to_csv(
            path +'/results_m_{}_act_{}_n_{}.csv'.format(m, str(act), str(batch_n)), index = True, header=results.keys())
        
        def strip_str(s):
            for i in range(len(s)):
                if s[i] == ' ':
                    s = s[:i-1] + '\\' + s[i:]
            return s
        # shell script that crops all plots
        # subprocess.call(['sh', './crop_figures.sh'])
        # subprocess.check_call(['./crop_plots_copy.sh', path])
        # subprocess.Popen(['./crop_plots.sh %s' % strip_str(path)], shell = True)


    except KeyboardInterrupt:
        import shutil
        shutil.rmtree(path)
