import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pysindy.differentiation import SmoothedFiniteDifference
import os

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

class find_governing_equations():

    def __init__(self, func, dt, dim, path) -> None:
        self.func = func
        self.dt = dt
        self.dim = dim
        self.path = path
        if not os.path.exists(path + "/Plots"):
            os.makedirs(path + "/Plots")

    
    def create_time_series(self, t_end, init):
        self.t_train = np.arange(0, t_end, self.dt)
        t_train_span = (self.t_train[0], self.t_train[-1])
        # initial condition as a list of two entries: [x1, x2]
        x0_train = init
        self.x_train = solve_ivp(self.func, t_train_span, x0_train,
                        t_eval=self.t_train, **integrator_keywords).y.T

    def find_equations(self, freq, deg, lamda = 0.1, plot = False, verbose=False):
        fourier_lib = ps.FourierLibrary(n_frequencies=freq)
        poly_lib = ps.PolynomialLibrary(degree=deg)
        identity_lib = ps.IdentityLibrary()
        diff_method = SmoothedFiniteDifference(smoother_kws={'window_length': 5})
        # opt = ps.optimizers.STLSQ(threshold=0.01, alpha = 0.1, max_iter=500, normalize_columns=True)
        opt = ps.optimizers.STLSQ(threshold=lamda, alpha = 0.1, 
                        max_iter=2000, normalize_columns=False, fit_intercept=False)

        # up to 3rd degree for van der pol
        # lib = ps.PolynomialLibrary(degree=3).fit(self.x_train)
        lib = ps.feature_library.GeneralizedLibrary([identity_lib, poly_lib, fourier_lib], 
                        library_ensemble=True).fit(self.x_train)

        model = ps.SINDy(optimizer=opt, 
                            feature_library=lib, 
                            differentiation_method=diff_method)
        model.fit(self.x_train, t=self.dt, multiple_trajectories=False)
        # Evolve the van der pol equations in time using a different initial condition
        t_test = np.arange(0, 20, self.dt)
        # x0_test = np.array([0, 1.5])
        x0_test = np.random.rand(1, self.dim)[0]
        t_test_span = (t_test[0], t_test[-1])
        x_test = solve_ivp(self.func, t_test_span, x0_test,
                        t_eval=t_test, **integrator_keywords).y.T
        # Compare SINDy-predicted derivatives with finite difference derivatives
        if verbose:
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

        return model

