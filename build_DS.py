import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pysindy.differentiation import SmoothedFiniteDifference
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
import time
import matplotlib

font = {'size'   : 13}

matplotlib.rc('font', **font)


# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'RK45'
integrator_keywords['atol'] = 1e-12

class find_governing_equations():

    def __init__(self, func, bounds, dt, t_end, dim, n_traj, path, verbose) -> None:
        self.func = func
        self.bounds = bounds
        self.dt = dt
        self.dim = dim
        self.path = path
        self.n_traj = n_traj
        self.t_train = np.arange(0, t_end, self.dt)
        self.t_train_span = (self.t_train[0], self.t_train[-1])
        self.verbose = verbose
        if not os.path.exists(path + "/Plots"):
            os.makedirs(path + "/Plots")

    def re(self, ls, thrsh, num):
        "Helper function for slicing time series"
        # slice array if it converged
        ct = 0
        for i in range(len(ls)):
            if np.abs(np.sum(ls[i])) < thrsh:
                ct +=1
            elif np.abs(np.sum(ls[i])) >= thrsh:
                ct = 0
            if ct >= num:
                return np.array(ls[:i-ct])
            # if it diverges, slice it
            if ((ls[i][1] > self.bounds[1]) or (ls[i][0] > self.bounds[0])):
                return np.array(ls[:i])
        return np.array(ls)

    
    def create_time_series(self, multiple = True):
        # initial condition as a list of two entries: [x1, x2]
        x0_train = [1, -0.5]
        self.x_train = solve_ivp(self.func, self.t_train_span, x0_train,
                        t_eval=self.t_train, **integrator_keywords).y.T
        if multiple == True:
            bounds = [-1,1]
            x0s = np.random.uniform(low=[ -x for x in bounds], high=bounds, size=(self.n_traj,self.dim))
            self.x_train_multi_sindy = []
            # x0s = np.random.rand(n_trajectories, dim)
            
            for i in range(self.n_traj):
                x_train_temp = solve_ivp(fun = self.func, t_span=self.t_train_span, 
                                            y0 = x0s[i], t_eval=self.t_train, **integrator_keywords).y.T
                self.x_train_multi_sindy.append(x_train_temp)
            


    def create_time_series_MLP(self, dim):
        x_train_multi = []
        y_train_multi = []
        x0s = np.random.uniform(low=[ -x for x in self.bounds], high=self.bounds, size=(self.n_traj,dim))

        for i in range(self.n_traj):
                x_train_temp = solve_ivp(fun = self.func, t_span=self.t_train_span, 
                                            y0 = x0s[i], t_eval=self.t_train, **integrator_keywords).y.T
                x_train_temp = self.re(x_train_temp, 1e-4, 5)
                # plt.figure()
                # plt.plot(x_train_temp, label = ['x1', 'x2'])
                # plt.title('x0: {:.2f}, {:.2f}'.format(x0s[i][0], x0s[i][1]))
                # plt.legend()
                                
                x_train_multi.append(x_train_temp)
        # now get finite difference approximations for each time series
        for x_train in x_train_multi:
            forward_diff = np.expand_dims(np.diff(x_train[:,0]) / self.dt, axis=1)
            forward_diff2 = np.expand_dims(np.diff(x_train[:,1]) / self.dt, axis=1)
            y_t = np.concatenate([forward_diff, forward_diff2], axis=1)
            y_train_multi.append(y_t)
        # slice first element to match y
        x_train_multi = [i[1:,:] for i in x_train_multi]
        return np.concatenate(x_train_multi), np.concatenate(y_train_multi)
    
    def process_MLP_inputs(self, x, y, batch_size = 1024):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True) #, random_state=100)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024)
        train_dataset = train_dataset.batch(batch_size)
        validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        validation_dataset = validation_dataset.batch(batch_size)
        return train_dataset, validation_dataset


    def find_equations(self, freq, deg, lamda = 0.1, plot = False):
        fourier_lib = ps.FourierLibrary(n_frequencies=freq)
        poly_lib = ps.PolynomialLibrary(degree=deg)
        identity_lib = ps.IdentityLibrary()
        diff_method = SmoothedFiniteDifference(smoother_kws={'window_length': 5})
        # opt = ps.optimizers.STLSQ(threshold=0.01, alpha = 0.1, max_iter=500, normalize_columns=True)
        opt = ps.optimizers.STLSQ(threshold=lamda, alpha = 0.1, 
                        max_iter=300, normalize_columns=False, fit_intercept=False)

        # up to 3rd degree for van der pol
        # lib = ps.PolynomialLibrary(degree=3).fit(self.x_train)
        lib = ps.feature_library.GeneralizedLibrary([identity_lib, poly_lib, fourier_lib], 
                        library_ensemble=True).fit(self.x_train)

        model = ps.SINDy(optimizer=opt, 
                            feature_library=lib, 
                            differentiation_method=diff_method)
        start_time = time.time()
        print('Fitting model...')
        if self.x_train_multi_sindy is None:
            model.fit(self.x_train, t=self.dt, multiple_trajectories=False)
        else:
            model.fit(self.x_train_multi_sindy, t=self.dt, multiple_trajectories=True)
        end_time = round(time.time() - start_time, 2)
        # Evolve the van der pol equations in time using a different initial condition
        t_test = np.arange(0, 20, self.dt)
        # x0_test = np.array([0, 1.5])
        x0_test = np.random.rand(1, self.dim)[0]
        t_test_span = (t_test[0], t_test[-1])
        x_test = solve_ivp(self.func, t_test_span, x0_test,
                        t_eval=t_test, **integrator_keywords).y.T
        # Compare SINDy-predicted derivatives with finite difference derivatives
        if self.verbose:
            model.print()
            print('Train Model score: %f' %model.score(self.x_train, self.dt))
            print('Test Model score: %f' % model.score(x_test, t=self.dt))
        if plot:
            # Predict derivatives using the learned model
            x_dot_test_predicted = model.predict(x_test)

            # Compute derivatives with a finite difference method, for comparison
            x_dot_test_computed = model.differentiate(x_test, t=self.dt)

            fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
            # fig.suptitle(
            #     'n: {}, degree: {}, modes: {}, $\lambda$: {}, train score: {:.3f}, test score: {:.3f}'.format(self.t_train.shape[0],
            #      deg, freq, lamda,model.score(self.x_train, self.dt), model.score(x_test, self.dt) ))
            for i in range(x_test.shape[1]):
                axs[i].plot(t_test, x_dot_test_computed[:, i],
                            'k', label='numerical derivative')
                axs[i].plot(t_test, x_dot_test_predicted[:, i],
                            'r--', label='model prediction')
                axs[i].legend()
                axs[i].set(xlabel='t', ylabel='$\dot x_{}$'.format(i))
            fig.tight_layout()
            fig.savefig(self.path + '/Plots/approximated_evolution.pdf')

            plt.figure(figsize = (9,5))
            plt.plot(x_dot_test_computed[:,0], x_dot_test_computed[:,1],'k', label='numerical derivative')
            plt.plot(x_dot_test_predicted[:,0], x_dot_test_predicted[:,1],'r--', label='model prediction')
            plt.xlabel('$\dot x_0$')
            plt.ylabel('$\dot x_1$')
            plt.legend()
            plt.savefig(self.path + '/Plots/approximated_phase_plot.pdf')
            plt.clf()
        self.model = model

        return model, end_time


    def find_equations_MLP(self, train_dataset, validation_dataset, plot_loss = True):
        l2_coeff = 1e-5
        reg = regularizers.l2(l2_coeff)
        act = 'relu'

        MLP_model = Sequential([
            Dense(20, activation=act, input_dim=2, kernel_regularizer = reg),
            Dense(256, activation=act, kernel_regularizer = reg),
            Dense(256, activation=act, kernel_regularizer = reg),
            Dense(512, activation=act, kernel_regularizer = reg),
            Dense(512, activation=act, kernel_regularizer = reg),
            Dense(512, activation=act, kernel_regularizer = reg),
            Dense(512, activation=act, kernel_regularizer = reg),
            Dense(256, activation=act, kernel_regularizer = reg),
            Dense(2)
        ])
        MLP_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=10)
        print('MLP training...')
        start_time = time.time()
        history = MLP_model.fit(train_dataset, validation_data=validation_dataset, epochs=15, verbose=self.verbose, callbacks=[earlystopping])
        MLP_end_time = round(time.time() - start_time, 2)
        if plot_loss:
            plt.figure(figsize=(10,7))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            # plt.plot(history.history['val_mae'])
            plt.title('Loss vs. epochs')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Training', 'Validation', 'Validation mae'])
            plt.savefig(self.path + '/Plots/MLP_Loss.pdf')
        self.MLP_model = MLP_model

        return MLP_model, MLP_end_time

    def evaluate_models(self, vf, n, plot_results = True):
        ls = np.stack([n, self.bounds],axis=1)

        def get_grid_of_points(lms):
            ls = [np.linspace(-i,i,int(n)) for n, i in lms]
            mesh_ls = np.meshgrid(*ls)
            all_mesh = [np.reshape(x, [-1]) for x in mesh_ls]
            grid_points = np.stack(all_mesh, axis=1)
            return grid_points
        data = get_grid_of_points(ls)

        y_pred = self.model.predict(data)
        y_pred_MLP = self.MLP_model.predict(data)
        y_true = np.array(vf(data)).T
        mse_sindy_x = np.mean(np.square(y_true - y_pred), axis=0)
        mse_mlp_x = np.mean(np.square(y_true - y_pred_MLP), axis=0)
        mse_sindy = np.mean(np.square(y_true - y_pred), axis=1)
        mse_mlp = np.mean(np.square(y_true - y_pred_MLP), axis=1)
        
        if plot_results:

            plt.figure(figsize=(15,7))
            plt.suptitle('SINDy test mse $x_1$: {:.3f},  SINDy test mse $x_2$: {:.3f}, MLP test mse $x_1$: {:.3f},  MLP test mse $x_2$: {:.3f}'.format(
                mse_sindy_x[0],mse_sindy_x[1], mse_mlp_x[0], mse_mlp_x[1]))
            plt.subplot(121)
            plt.plot(y_true[:,0], label = 'True')
            plt.plot(y_pred[:,0], label = 'SINDy Predicted')
            plt.plot(y_pred_MLP[:,0], label = 'MLP Predicted')
            plt.title('$x_1$')
            plt.legend()
            plt.subplot(122)
            plt.plot(y_true[:,1], label = 'True')
            plt.plot(y_pred[:,1], label = 'SINDy Predicted')
            plt.plot(y_pred_MLP[:,1], label = 'MLP Predicted')
            plt.title('$x_2$')
            plt.legend()
            plt.savefig(self.path + '/Plots/function_evaluations_dt_{}_traj_{}.pdf'.format(self.dt, self.n_traj))


            plt.figure(figsize=(15,7))
            plt.subplot(121)
            plt.imshow(mse_sindy.reshape(n[0],n[1]), cmap='gnuplot')
            plt.title('SINDy')
            # plt.legend()
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(mse_mlp.reshape(n[0],n[1]), cmap='gnuplot')
            plt.title('MLP')
            # plt.legend()
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(self.path + '/Plots/function_mse_dt_{}_traj_{}.pdf'.format(self.dt, self.n_traj))
        return mse_sindy_x[0], mse_sindy_x[1], mse_mlp_x[0], mse_mlp_x[1]
