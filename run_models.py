from build_DS import find_governing_equations
from get_vector_field import get_equation
from NN_Lyapunov import build_lyapunov, LyapunovModel
import tensorflow as tf
import numpy as np
import os
import subprocess
import pandas as pd




if __name__ == "__main__":

    try: 
        hidden_u = []
        m = 200
        dim = 2
        bounds = [1.6, 4]
        n_x = 10
        n_y = 20
        batch_n = n_x*n_y
        buff = None
        epochs = 2000
        tol = 1e-3
        act = tf.math.cos
        # act = 'relu'
        train = True
        if not train:
            lr = m * 0.1
            opt = tf.keras.optimizers.SGD(learning_rate=lr, nesterov=True)
        else:
            lr = 0.03
            opt = tf.keras.optimizers.Adam(lr)
        use_true_equation = True
        # time series hyperparameters
        dt = 0.01
        init= [0, 2]
        t_end = 20
        eq = 'van_der_pool_2d'
        vf = get_equation(eq)

        params = {'eq': eq, 'm': m, 'dim': dim, 'bounds0': bounds[0], 'bounds1': bounds[1], 'n_x': n_x, 'n_y': n_y, 'batch_n': batch_n, 'epochs': epochs, 'tol': tol,
                    'activation': str(act), 'train_params': train, 'opt':  str(opt.get_slot_names)[:-27][-3:],  'lr': lr, 'use_true_equation': use_true_equation,
                    'dt': dt, 'init0': init[0], 'init1': init[1], 't_end': t_end}

        if not os.path.exists("Experiments"):
            os.makedirs("Experiments")

        path = 'Experiments/Lyapunov_eq_{}_train_{}'.format(eq, train)

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

        with open(path + "/model_params.txt", 'w') as f: 
            for key, value in params.items(): 
                f.write('%s:%s\n' % (key, value))

        # insert kernel here aswell

        if not use_true_equation:
            GE = find_governing_equations(func = get_equation('van_der_pool_1d'), dt= dt)
            GE.create_time_series(t_end = t_end, init = init)
            model = GE.find_equations(verbose = True)

            def vf(x):
                if type(x) != np.ndarray:
                    x = x.numpy()
                return model.predict(x).T

        base_model = build_lyapunov(vf, dim, bounds, m, epochs, tol, path)

        train_dataset, train_data_points, train_input_RHS = base_model.create_dataset(n_x, n_y, 
                                        dim, bounds, batch_n, buff = None, train = True, plot=True)
        test_dataset, test_data_points, test_input_RHS = base_model.create_dataset(n_x, n_y, 
                                        dim, bounds, batch_n, buff = None, train = False, plot=True)
        my_mlp = base_model.get_regularised_bn_mlp(act, hidden_u, LyapunovModel, opt= opt,  train=train)
        # print(my_mlp.summary())
        base_model.plot_Layer(20)
        all_loss_values, all_test_loss_values, fitted_model = base_model.fit(train_dataset, test_dataset, plot=True)
        fitted_model.save(path + '/Lyapunov_{}d_{}m_{}epochs_opt_{}_lr_{}_eq_{}.h5'.format(dim, m,epochs, str(opt.get_slot_names)[:-27][-3:], lr, eq))
        Zp, Ze = base_model.plot_solution(20)
        base_model.plot_Layer(20)

        results = {'all_loss_values': all_loss_values, 'all_test_loss_values': all_test_loss_values,
                        'Zp': Zp, 'Ze': Ze}
        
        pd.DataFrame.from_dict(data=results).to_csv(path +'/results.csv', header=False)
        

        # # shell script that crops all plots
        # subprocess.call(['sh', './crop_plots.sh'])

    except KeyboardInterrupt:
        import shutil
        shutil.rmtree(path)