"""
Main program that runs the Lyapunov functions approximation 
using Neural Networks
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import os 
from mpl_toolkits.mplot3d import Axes3D
import sys
import shutil
# checking if the directory Plots
# exist or not.
if not os.path.exists("Plots"):
    os.makedirs("Plots")



class LyapunovModel(tf.keras.Model):
    """
    This class defines the custom Tensorflow model used for the Experiments. 
    It includes custom training and testing steps and custom fitting procedures.
    """
    def __init__(self, path, **kwargs):
        super(LyapunovModel, self).__init__(**kwargs)
        self.path = path

    def get_optimizer(self, opt):
        # Defined here because @tf.function train_step does not allow self.opt
        self.opt = opt

    def train_step(self, x_batch_train, vf_batch_train):
        """
        x_batch_train: (n,2)
        vf_batch_train: (n,2)
        """
        # start gradient recording for derivative w.r.t. model parameters
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tapex:
                # prepare for evaluation of x-derivative
                tapex.watch(x_batch_train)
                logits2 = self(x_batch_train, training = True) # (n, 1)

            # evaluate x-derivative
            gradx = tapex.gradient(logits2, x_batch_train) # (n,2)

            # evaluate PDE part of loss function

            inner_product = tf.reduce_sum(gradx*vf_batch_train, axis=1) # (n,)
            target = tf.square(tf.norm(x_batch_train, ord = 2, axis=1)) # (n,)
            individual_loss = tf.square(tf.abs((inner_product + target))) # (n,)
            loss_value = tf.reduce_sum(individual_loss) / len(x_batch_train) # scalar

        # evaluate derivative w.r.t. model parameters
        # changed this
        grads = tape.gradient(loss_value, self.trainable_weights) # (m, 1)

        # Run one step of gradient descent optimizer
        self.opt.apply_gradients(zip(grads, self.trainable_weights))

        return(loss_value)

    def test_step(self, x_test, vf_test):

        with tf.GradientTape() as tapex:

            tapex.watch(x_test)
            logits2 = self(x_test, training = False)

        # evaluate x-derivative
        gradx = tapex.gradient(logits2, x_test)
        inner_product = tf.reduce_sum(gradx*vf_test, axis=1) # (n,)
        target = tf.square(tf.norm(x_test, ord = 2, axis=1)) # (n,)
        individual_loss = tf.square(tf.abs((inner_product + target))) # (n,)
        loss_value = tf.reduce_sum(individual_loss) / len(x_test) # scalar
        return loss_value


    def my_fit(self, epochs, tol, train_dataset, validation_dataest):
        """
        Custom fitting procedure. Only prints 5 messages while the network trains.
        """
        all_loss_values = []
        all_test_loss_values = []

        # loop through the epochs
        for epoch in range(epochs):
            # initialize error variable, mlv=maximum (L_infty)
            mlv = 0.0
            
            # Iterate over the batches of the dataset
            # in our case only 1 batch so inly 1 loop here
            all_batch_loss_values = []
            for _, (x_batch_train, vf_batch_train) in enumerate(train_dataset):

                # call optimization routine
                loss_value = self.train_step(x_batch_train, vf_batch_train)
                all_batch_loss_values.append(loss_value.numpy())
                if loss_value.numpy() > 1000000:
                    print('Loss exploded')
                    shutil.rmtree(self.path)
                    sys.exit()

                mlv = tf.reduce_max([mlv, loss_value], axis =0)
                if mlv < tol:
                    break
            all_loss_values.append(np.array(all_batch_loss_values).mean())
            
            all_batch_loss_test_values = []
            for _, (x_test, vf_test) in enumerate(validation_dataest):
                test_loss = self.test_step(x_test, vf_test)
                all_batch_loss_test_values.append(test_loss.numpy())
            all_test_loss_values.append(np.array(all_batch_loss_test_values).mean())


            if epoch % int(epochs/5) == 0:
                print('epoch %2s, train loss %10.6f, test loss %10.6f' % (epoch, float(loss_value), float(test_loss)))
                
        return all_loss_values, all_test_loss_values

class build_lyapunov():
    """
    This class takes any input function and computes the Lyapunov function approximation
    """

    def __init__(self, func, dim, act, bounds, m, epochs, tol, path, func_name) -> None:
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.m = m
        self.epochs = epochs
        self.tol = tol
        self.path = path
        self.act = act
        self.func_name = func_name
        if not os.path.exists(path + "/Plots"):
            os.makedirs(path + "/Plots")


    def create_dataset(self, n, dim, bounds, batch_n, buff = None, train = True, plot=False):
        """
        Creates the dataset we will be using
        """
        if train == False:
            data = np.random.uniform(low=[ -x for x in bounds], high=bounds, size=(np.prod(n),dim))
        else:
            ls = np.stack([n, bounds],axis=1)
            def get_grid_of_points(lms):
                "helper function that creates grid for any dimensions"
                ls = [np.linspace(-i,i,int(n)) for n, i in lms]
                mesh_ls = np.meshgrid(*ls)
                all_mesh = [np.reshape(x, [-1]) for x in mesh_ls]
                grid_points = np.stack(all_mesh, axis=1)
                return grid_points
            data = get_grid_of_points(ls)
            
        data_points = tf.constant(data, tf.float32) # (n[1], dim)
                        
        input_RHS =  tf.transpose(tf.convert_to_tensor(self.func(data_points), dtype=tf.float32))  # (n[1], dim)

        dataset_raw = tf.data.Dataset.from_tensor_slices((data_points, input_RHS))
        if buff:
            dataset_raw = dataset_raw.shuffle(buffer_size=buff)
        dataset = dataset_raw.batch(batch_n)

        if plot:
            plt.scatter(data_points[:,0], data_points[:,1], label = str(train))
            plt.savefig(self.path + '/Plots/dataset_train_{}.pdf'.format(train))
            plt.clf()

        return dataset, data_points, input_RHS

    def get_regularised_bn_mlp(self,hidden_units, keras_model, opt, train=False):
        """
        This function is used to build the MLP model. It takes input_shape and hidden_units
        as arguments, which should be used to build the model as described above, using the
        functional API.
        Your function should return the model.
        """
        inputs = tf.keras.layers.Input(shape=(self.dim,), name = 'state')
        h = tf.keras.layers.Dense(self.m, activation=self.act, 
                                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1),
                                bias_initializer= tf.random_uniform_initializer(0, 2*  np.pi), 
                                trainable=train, name = '1st_hidden')(inputs)
        h = np.sqrt(2) * h 
        for units in hidden_units:
            h = tf.keras.layers.Dense(units, activation='linear')(h)
        # use bias is false because the derivative of a constant number (bias) is just zero
        outputs = tf.keras.layers.Dense(1, kernel_initializer = tf.zeros_initializer(),activation=None, name = 'output_layer', use_bias=False, trainable=True)(h)
        outputs = outputs / self.m 
        model = keras_model(path = self.path, inputs=inputs, outputs=outputs)
        model.get_optimizer(opt)
        self.model = model
        return model

    def plot_Layer(self, numpoints):
        """
        Simply plot the Layer without any training to 
        see its initial state and compare with actual Gaussian kernel
        """
        x = np.linspace(-self.bounds[0], self.bounds[0], numpoints)
        y = np.linspace(-self.bounds[1], self.bounds[1], numpoints)

        X, Y = np.meshgrid(x, y)

        s = X.shape
        DT = np.zeros((numpoints**2,self.dim))

        # convert mesh into point vector for which the model can be evaluated
        c = 0
        for i in range(s[0]):
            for j in range(s[1]):
                DT[c,0] = X[i,j]
                DT[c,1] = Y[i,j]
                c = c+1;

        intermediate_output1 = tf.keras.Model(self.model.input, self.model.get_layer('1st_hidden').output)

        x1 = tf.zeros(DT.shape, tf.float32) # (n**2, 2)
        x2 = tf.constant(DT, tf.float32) # (n**2, 2)
        phi_1 = intermediate_output1(x1) # (n**2, m)
        phi_2 = intermediate_output1(x2) # (n**2, m)
        k_nn = tf.reduce_sum(2 * phi_1*phi_2, axis=1) / self.m # (n**2,) Eq. 22 paper
        k_nn = tf.reshape(k_nn, (numpoints, numpoints)) # (n, n)

        g_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic()
        kernel_sol = g_kernel.apply(x1, x2) # (n**2,)
        kernel_sol = tf.reshape(kernel_sol, (numpoints,numpoints)) # (n,n)

        plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.title('Actual Gasussian Kernel')
        cp1 = plt.contour(y,y, kernel_sol, levels=10);
        plt.clabel(cp1, inline=True, fontsize=10)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')

        plt.subplot(122)
        plt.title('Neural Network Kernel')
        cp2 = plt.contour(y,y, k_nn, levels=10)
        plt.clabel(cp2, inline=True, fontsize=10)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')

        plt.savefig(self.path + '/Plots/NN_Kernel_m_{}_{}.pdf'.format(self.m, str(self.act)))
        plt.clf()

    def fit(self, train_dataset, test_dataset, plot = False):
        # uses custom fit from the LyapunovModel class
        all_loss_values, all_test_loss_values = self.model.my_fit(self.epochs, self.tol, train_dataset, test_dataset)
        if plot:
            plt.plot(all_loss_values, label = 'train loss')
            plt.plot(all_test_loss_values, label = 'validation loss')
            plt.title("Loss vs iterations with m = {}".format(self.m))
            plt.xlabel("Iterations")
            plt.ylabel("MSE loss")
            plt.legend()
            plt.savefig(self.path + '/Plots/Loss_Curve_m_{}_act_{}.pdf'.format(self.m, str(self.act)))
            plt.clf()
        return all_loss_values, all_test_loss_values, self.model

    def plot_solution(self, numpoints):
        """
        Evaluates the results on a new testing dataset. Only works for 2D.  Returns MSE_{od}
        """

        # define plotting range and mesh
        x = np.linspace(-self.bounds[0], self.bounds[0], numpoints)
        y = np.linspace(-self.bounds[1], self.bounds[1], numpoints)

        X, Y = np.meshgrid(x, y)

        s = X.shape

        Ze = np.zeros(s)
        Zp = np.zeros(s)
        tar = np.zeros(s)
        DT = np.zeros((numpoints**self.dim,self.dim))

        # convert mesh into point vector for which the model can be evaluated
        c = 0
        for i in range(s[0]):
            for j in range(s[1]):
                DT[c,0] = X[i,j]
                DT[c,1] = Y[i,j]
                c = c+1;

        # evaluate model (= Lyapunov function values V)
        Ep = self.model.predict(DT)
        # convert point vector to tensor for evaluating x-derivative
        tDT = tf.convert_to_tensor(DT, dtype=tf.float32)

        # evaluate gradients DV of Lyapunov function
        with tf.GradientTape() as tape:
            tape.watch(tDT)
            ypm = self.model(tDT)
            grads = tape.gradient(ypm, tDT)

        # compute orbital derivative DVf
        Ee = tf.math.reduce_sum(grads*tf.transpose(tf.convert_to_tensor(self.func(DT), dtype=tf.float32)), axis=1)
        target = -tf.square(tf.norm(tDT, ord = 2, axis=1))
        mse = tf.keras.metrics.mean_squared_error(target, Ee)
        # copy V and DVf values into plottable format
        c = 0
        for i in range(s[0]):
            for j in range(s[1]):
                Ze[i,j] = Ee[c]
                Zp[i,j] = Ep[c]
                tar[i,j] = target[c]
                c = c+1;


        # define vector field for plot
        new_vf = np.reshape(np.array(self.func(DT)), (self.dim, numpoints, numpoints))

        ##################################################################################

        # define figure
        fig = plt.figure(figsize=(10,10))
        # ax = plt.axes(projection='3d')
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        # ax.set_title('m = {}'.format(m))
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('DV');

        # plot values V
        ax.plot_surface(X, Y, tar, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        # # plot values V
        # ax.plot_surface(X, Y, Zp, rstride=1, cstride=1,
        #                 cmap='magma_r', edgecolor='none', alpha = 0.8)

        # plot orbital derivative DVf
        ax.plot_wireframe(X, Y, Ze, rstride=1, cstride=1)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('DV');

        # plot values V
        ax.plot_surface(X, Y, tar, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        # # plot values V
        # ax.plot_surface(X, Y, Zp, rstride=1, cstride=1,
        #                 cmap='magma_r', edgecolor='none')

        # plot orbital derivative DVf
        ax.plot_wireframe(X, Y, Ze, rstride=1, cstride=1)

        ax.view_init(20, 180)

        plt.savefig(self.path + '/Plots/{}_plot_m_{}_act_{}.pdf'.format(self.func_name, self.m, str(self.act)))
        # plt.show()

        ##################################################################################

        plt.figure(figsize=(5,8))
        cp = plt.contour(x,y,Zp, levels=15)
        plt.quiver(X,Y, new_vf[0], new_vf[1], headwidth=2, headlength=4)
        plt.clabel(cp, inline=True, fontsize=10)
        plt.title('m = {}'.format(self.m))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.savefig(self.path + '/Plots/{}_m_{}_act_{}.pdf'.format(self.func_name, self.m, str(self.act)))

        # plt.show()
        # return Zp, Ze
        return mse.numpy()


    def plot_solution_ND(self, numpoints):
        """
        Evaluates the results on a new testing dataset for any number of dimensions. Returns MSE_{od}
        """

        ls = np.stack([numpoints, self.bounds],axis=1)

        def get_grid_of_points(lms):
            ls = [np.linspace(-i,i,int(n)) for n, i in lms]
            mesh_ls = np.meshgrid(*ls)
            all_mesh = [np.reshape(x, [-1]) for x in mesh_ls]
            grid_points = np.stack(all_mesh, axis=1)
            return grid_points
        data = get_grid_of_points(ls)

        # convert point vector to tensor for evaluating x-derivative
        tDT = tf.convert_to_tensor(data, dtype=tf.float32)

        # evaluate gradients DV of Lyapunov function
        with tf.GradientTape() as tape:
            tape.watch(tDT)
            ypm = self.model(tDT)
            grads = tape.gradient(ypm, tDT)

        # compute orbital derivative DVf
        Ee = tf.math.reduce_sum(grads*tf.transpose(tf.convert_to_tensor(self.func(data), dtype=tf.float32)), axis=1)
        target = -tf.square(tf.norm(tDT, ord = 2, axis=1))
        mse = tf.keras.metrics.mean_squared_error(target, Ee)
        return mse.numpy()

    def psnd(self, numpoints, g, q):
        """
        Plots the Lyapunov function and its orbital derivative for any dimension. 
        Make sure that dimensions g, q are within the dimensions of the equation
        """
        # define plotting range and mesh
        x = np.linspace(-1, 1, numpoints)
        y = np.linspace(-1, 1, numpoints)

        X, Y = np.meshgrid(x, y)

        s = X.shape

        Ze = np.zeros(s)
        Zp = np.zeros(s)
        DT = np.zeros((numpoints**2,self.dim))

        # convert mesh into point vector for which the model can be evaluated
        c = 0
        for i in range(s[0]):
            for j in range(s[1]):
                DT[c,g] = X[i,j]
                DT[c,q] = Y[i,j]

                c = c+1;

        Ep = self.model.predict(DT)

        # convert point vector to tensor for evaluating x-derivative
        tDT = tf.convert_to_tensor(DT, dtype=tf.float32)

        # evaluate gradients DV of Lyapunov function
        with tf.GradientTape() as tape:
            tape.watch(tDT)
            ypm = self.model(tDT)
            grads = tape.gradient(ypm, tDT)

        # compute orbital derivative DVf
        Ee = tf.reduce_sum(grads*tf.transpose(tf.convert_to_tensor(self.func(DT), dtype=tf.float32)), axis=1)
            
        # copy V and DVf values into plottable format
        c = 0
        for i in range(s[0]):
            for j in range(s[1]):
                Ze[i,j] = Ee[c]
                Zp[i,j] = Ep[c]
                c = c+1;

        # define figure
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('$x_{}$'.format(g),fontsize=13)
        ax.set_ylabel('$x_{}$'.format(q),fontsize=13)
        ax.set_zlabel('$DWf, W$',fontsize=13);
        ax.plot_surface(X, Y, Zp, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        # plot orbital derivative DVf
        ax.plot_wireframe(X, Y, Ze, rstride=1, cstride=1)
        plt.savefig(self.path + '/Plots/dww_{}_{}.pdf'.format(g,q))