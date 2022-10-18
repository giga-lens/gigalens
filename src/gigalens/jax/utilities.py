import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
from corner import corner
import jax
import numpy as np


tfd = tfp.distributions


def initialize_default_prior(theta_E = 1.25):
    
    lens_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    theta_E=tfd.LogNormal(jnp.log(theta_E), 0.25),
                    gamma=tfd.TruncatedNormal(2, 0.25, 1, 3),
                    e1=tfd.Normal(0, 0.1),
                    e2=tfd.Normal(0, 0.1),
                    center_x=tfd.Normal(0, 0.05),
                    center_y=tfd.Normal(0, 0.05),
                )
            ),
            tfd.JointDistributionNamed(
                dict(gamma1=tfd.Normal(0, 0.05), gamma2=tfd.Normal(0, 0.05))
            ),
        ]
    )
    
    lens_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
                    n_sersic=tfd.Uniform(2, 6),
                    e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                    e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                    center_x=tfd.Normal(0, 0.05),
                    center_y=tfd.Normal(0, 0.05),
                    Ie=tfd.LogNormal(jnp.log(500.0), 0.3),
                )
            )
        ]
    )

    source_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15),
                    n_sersic=tfd.Uniform(0.5, 4),
                    e1=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
                    e2=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
                    center_x=tfd.Normal(0, 0.25),
                    center_y=tfd.Normal(0, 0.25),
                    Ie=tfd.LogNormal(jnp.log(150.0), 0.5),
                )
            )
        ]
    )

    prior = tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior]
    )
    
    return prior


def plot_residual(observed_img,
                  best_fit,
                  lens_sim, 
                  mask_img = None, 
                  background_rms = 0.2, 
                  exp_time = 100,
                  norm = None,
                  vmin = 0,
                  vmax = 20,
                  ):
    
    if norm is None:
        norm = mpl.colors.PowerNorm(0.5, vmin=vmin, vmax=vmax)
        
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    sim_img = jnp.array(lens_sim.simulate(best_fit))
    
    if mask_img is not None:
        observed_img = observed_img * mask_img
        sim_img = sim_img * mask_img
    
    img0 = ax[0].imshow(observed_img, norm=norm,)
    img1 = ax[1].imshow(sim_img, norm=norm)
    plt.colorbar(img0, ax = ax[0])
    plt.colorbar(img1, ax = ax[1])
    resid = sim_img - observed_img
    err_map = jnp.sqrt(background_rms**2 + sim_img/exp_time)
    img2 = ax[2].imshow(resid/err_map, cmap='coolwarm', interpolation='none', vmin=-5, vmax=5)
    plt.colorbar(img2, ax = ax[2], label = r'$ \frac{f_{obs} - f_{sim}}{\sqrt{\mathrm{f_{sim}/exp} + \mathrm{bkg}^2}} $')
    pull = resid/jnp.std(resid)
    img3 = ax[3].imshow(pull, cmap='coolwarm', interpolation='none', vmin=-5, vmax=5)
    plt.colorbar(img3, ax = ax[3], label = r'$ \frac{f_{obs} - f_{sim}}{\sigma} $')


    ax[0].set_title("Observed Image")
    ax[1].set_title("Simulated Image")
    ax[2].set_title("Errormap Residual")
    ax[3].set_title("Difference Residual (Pull)")
    
    return fig, ax
    
def get_med_solution(prob_model, samples):
    """Only works for EPL + Shear Currently."""
    med_lens = []
    med_shear = []
    med_ll = []
    med_sl = []

    lens_samp = get_samples(prob_model.bij.forward(samples))
    ll_samp = get_samples_light(prob_model.bij.forward(samples), 1)
    sl_samp = get_samples_light(prob_model.bij.forward(samples), 2)

    for i in range(lens_samp.shape[0]):
        med_val = jnp.median(lens_samp[i, :, :].flatten())
        if i  < lens_samp.shape[0] - 2:
            med_lens.append(med_val)
        else:
            med_shear.append(med_val)

    for i in range(ll_samp.shape[0]):
        med_val = jnp.median(ll_samp[i, :, :].flatten())
        med_ll.append(med_val)

    for i in range(sl_samp.shape[0]):
        med_val = jnp.median(sl_samp[i, :, :].flatten())
        med_sl.append(med_val)
    
    return truth_format([med_lens, med_shear, med_ll, med_sl])

def truth_format(a):
    mass, massg = {'theta_E':None, 'gamma': None, 'e1': None, 'e2':None, 'center_x':None, 'center_y': None}, {'gamma1':None, 'gamma2':None}
    llight = {'R_sersic': None, 'n_sersic': None, 'e1': None, 'e2':None, 'center_x':None, 'center_y': None, 'Ie': None}
    slight = {'R_sersic': None, 'n_sersic': None, 'e1': None, 'e2':None, 'center_x':None, 'center_y': None, 'Ie': None}

    for i, lbl in enumerate(mass.keys()):
        if i <= 5:
            mass[lbl] = a[0][i]

    for i, lbl in enumerate(massg.keys()):
            massg[lbl] = a[1][i]

    for i, lbl in enumerate(llight.keys()):
            llight[lbl] = a[2][i]

    for i, lbl in enumerate(slight.keys()):
            slight[lbl] = a[3][i]
            
    return [[mass, massg],[llight],[slight]]

def get_samples_lens(x):
    return jnp.array(
                [
                    x[0][0]['theta_E'],
                    x[0][0]['gamma'],
                    x[0][0]['e1'],
                    x[0][0]['e2'],
                    x[0][0]['center_x'],
                    x[0][0]['center_y'],
                    x[0][1]['gamma1'],
                    x[0][1]['gamma2'],
                ]
            )

def get_samples_light(x, i):
    return jnp.array(
                    [
                        x[i][0]['R_sersic'],
                        x[i][0]['n_sersic'],
                        x[i][0]['e1'],
                        x[i][0]['e2'],
                        x[i][0]['center_x'],
                        x[i][0]['center_y'],
                        x[i][0]['Ie']
                    ]
                )

def generate_cornerplot_lens(physical_samples):
    return corner(physical_samples.reshape((8,-1)).T,
                  show_titles=True, title_fmt='.3f',
                  labels=[r'$\theta_E$', r'$\gamma$', r'$\epsilon_1$', r'$\epsilon_2$', r'$x$', r'$y$', r'$\gamma_{1,ext}$', r'$\gamma_{2,ext}$']);

def generate_cornerplot_light(physical_samples):
    return corner(physical_samples.reshape((7,-1)).T,
                  show_titles=True, title_fmt='.3f',
                  labels=[r'$R_{ss}$', r'$n_{ss}$', r'$\epsilon_1$', r'$\epsilon_2$', r'$x$', r'$y$', r'$I_e$'])

def ellip_mask(x_center, y_center, a, b, num_pix, theta):
    """ x_center and y_center are relative to the image center. """
    
    nx = num_pix   # number of pixels in x-dir
    ny = num_pix   # number of pixels in y-dir

    # set up a coordinate system
    x = jnp.linspace(-num_pix / 2, num_pix / 2, nx)
    y = jnp.linspace(-num_pix / 2, num_pix / 2, ny)

    # Setup arrays which just list the x and y coordinates
    x_grid, y_grid = jnp.meshgrid(x, y)
    
    xgrid = (x_grid - x_center) * jnp.cos(theta) + (y_grid - y_center) * jnp.sin(theta)
    ygrid = -(x_grid - x_center) * jnp.sin(theta) + (y_grid - y_center) * jnp.cos(theta) 

    # Calculate the ellipse values all at once
    ellipse = (xgrid) **2 / a**2 + (ygrid) **2 / b**2

    # Create an array of int32 zeros
    grey = jnp.ones((nx,ny), dtype=jnp.int32)

    # Put 1's where ellipse is less than 1.0
    # Note ellipse <1.0 produces a boolean array that then indexes grey
    grey = grey.at[ellipse <= 1.0].set(0)

    return grey

def check_priors(prior, best_fit, truth = None, seed = 100):
    fig, ax = plt.subplots(1, 1, figsize = (18, 6), squeeze = 0)
    _key = jax.random.PRNGKey(seed)
    sample = prior.sample(1000, _key)
    i = 0
    j = 0
    
    ax[i, j].axhline(0, c = "k", ls = "--")
    ax[i, j].axhline(1, c = "r", ls = "--")
    ax[i, j].axhline(-1, c = "r", ls = "--")

    params = []
    res = []

    for mod in best_fit:
        for parameter in mod:
            k = np.array(list(parameter.keys()))
            v = np.array(list(parameter.values()))
            idx = np.argsort(k)
            params.extend(k[idx])
            res.extend(v[idx])

    if truth is not None:
        true_params = []
        true_res = []

        for mod in truth:
            for parameter in mod:
                k = np.array(list(parameter.keys()))
                v = np.array(list(parameter.values()))
                idx = np.argsort(k)
                true_params.extend(k[idx])
                true_res.extend(v[idx])


    samp_params = []
    dev = []
    mn = []

    for mod in sample:
        for parameter_set in mod:
            mean = []
            stddev = []
            for parameter in parameter_set.values():
                mean.append(parameter.mean())
                stddev.append(parameter.std())
            mean = np.array(mean)
            stddev = np.array(stddev)
            k = np.array(list(parameter_set.keys()))
            idx = np.argsort(k)
            samp_params.extend(k[idx])
            dev.extend(stddev[idx])
            mn.extend(mean[idx])

    res = np.array(res)
    ax[i, j].scatter(range(len(res)), (res - mn) / dev, label = "Result As Std off Prior")
    if truth is not None:
        true_res = np.array(true_res)
        ax[i, j].scatter(range(len(res)), (true_res - mn) / dev , label = "Truth As Std off Prior")
    ax[i,j].set_xticks(range(len(params)))
    ax[i,j].set_xticklabels(params, rotation = 60)
    ax[i,j].legend()
    ax[i, j].set_ylabel("Standard Deviations off of Prior")

    fig.suptitle("Truth and Result Compared to Prior", fontsize = 15, y = 0.94)
    plt.show()

    if truth is not None:
        print("")
        print("{:10}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}".format("Param", "InitMn", "InitDev", "Truth", "Result", "ResSig", "TrthSig"))
        print("\n".join("{:10}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}".format(a, x, y, z, q, w, s) 
                        for a, x, y, z, q, w, s 
                        in zip(params, mn, dev, true_res, res, (res - mn)/dev, (true_res - mn)/dev)))
    else:
        print("")
        print("{:10}\t{:>8}\t{:>8}\t{:>8}\t{:>8}".format("Param", "InitMn", "InitDev", "Result", "ResSig"))
        print("\n".join("{:10}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}".format(a, x, y, z, q) 
                        for a, x, y, z, q
                        in zip(params, mn, dev, res, (res - mn)/dev)))
    
    return fig, ax

def ellip_mask_xy(x, y, a, b, num_pix, theta):
    """ x and y are pixel positions in the image. """
    
    
    nx = num_pix   # number of pixels in x-dir
    ny = num_pix   # number of pixels in y-dir

    # set up a coordinate system
    x = jnp.linspace(-num_pix / 2, num_pix / 2, nx)
    y = jnp.linspace(-num_pix / 2, num_pix / 2, ny)

    # Setup arrays which just list the x and y coordinates
    x_grid, y_grid = jnp.meshgrid(x, y)
    
    xgrid = (x_grid - x_center) * jnp.cos(theta) + (y_grid - y_center) * jnp.sin(theta)
    ygrid = -(x_grid - x_center) * jnp.sin(theta) + (y_grid - y_center) * jnp.cos(theta) 

    # Calculate the ellipse values all at once
    ellipse = (xgrid) **2 / a**2 + (ygrid) **2 / b**2

    # Create an array of int32 zeros
    grey = jnp.ones((nx,ny), dtype=jnp.int32)

    # Put 1's where ellipse is less than 1.0
    # Note ellipse <1.0 produces a boolean array that then indexes grey
    grey = grey.at[ellipse <= 1.0].set(0)

    return grey


def square_mask(x_low, x_high, y_low, y_high, num_pix):
    
    mask_img = jnp.ones((num_pix, num_pix))
    mask_img[x_low:x_high, y_low:y_high] = 0
    
    return mask_img
    
    