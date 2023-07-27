import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from functional_gp_toolbox import kernel_tools 
from functional_gp_toolbox import GP_regression_tools

k_tools = kernel_tools()
#kernel = k_tools.get_polynomial_kernel(order = rnd.choice(range(0,9)))
#kernel = k_tools.get_squared_exponential_kernel(length_scale = np.random.uniform(low = 0.05, high = 0.5))
kernel = k_tools.get_periodic_kernel(frequency = np.random.uniform(low = 0.2, 
                                                                   high = 5))

number_functions = 1
function_length = 200
function_range = np.linspace(-1,1,function_length)

number_samples = 30
sample_indices = np.sort(np.random.choice(range(0,function_length/4), 
                                          size = (number_samples,), 
                                          replace = "no")).tolist()
sample_range = function_range[sample_indices]

noise_level = 0.5
functions = k_tools.get_function_from_kernel(kernel, 
                                             number_functions, 
                                             function_range)
corrupted_samples = k_tools.get_corrupted_samples(functions, 
                                                  sample_indices,
                                                  noise_level = noise_level)
k_tools.plot_corrupted_samples(function_range, 
                               functions, 
                               sample_range, 
                               corrupted_samples)
plt.xlabel("Time (s)")

## Optimal GP analyss ##
GP_tools = GP_regression_tools()
GP_posterior = GP_tools.get_GP_regression_posterior(data_points = corrupted_samples[0], 
                                                    kernel = kernel, 
                                                    data_range = sample_range, 
                                                    noise_level = noise_level)
target_range = np.arange(-1.,1.,0.005)
k_tools.plot_corrupted_samples(function_range, 
                               functions, 
                               sample_range, 
                               corrupted_samples)
GP_tools.plot_GP_posterior(GP_posterior_expectation = GP_posterior["expectation"], 
                           GP_posterior_variance = GP_posterior["variance"],
                           target_range = target_range)
plt.title("Optimal GP analysis")
plt.xlabel("Time (s)")

## Optimal GP analysys (PDF) ##
value_range = np.linspace(-3.,3.,100)
k_tools.plot_corrupted_samples(function_range, 
                               functions, 
                               sample_range, 
                               corrupted_samples,
                               plot_samples = "no")
GP_tools.plot_posterior_pdf(GP_posterior["pdf"], value_range, target_range)
plt.title("Optimal GP posterior densities")
plt.xlabel("Time (s)")

# multiple GP analysis
freq_list = np.arange(0.2,5.,0.1)
kernels_list = [k_tools.get_periodic_kernel(frequency = l) for l in freq_list]
results_list = GP_tools.multiple_GP_analysis(data_points = corrupted_samples[0], 
                                             kernels_list = kernels_list, 
                                             data_range = sample_range, 
                                             noise_level = noise_level)

# Model posterior probabilities

meta_results = GP_tools.meta_GP_analysis(results_list)
plt.plot(np.array(freq_list), meta_results["probabilities"])
plt.ylim(0.,0.8)
plt.xlim(min(freq_list),max(freq_list))
plt.title("Model posterior probabilities")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Probability")

## Optimized posterior analysis ##
k_tools.plot_corrupted_samples(function_range, 
                               functions, 
                               sample_range, 
                               corrupted_samples)
GP_tools.plot_GP_posterior(GP_posterior_expectation = meta_results["optimized"]["expectation"], 
                           GP_posterior_variance = meta_results["optimized"]["variance"],
                           target_range = target_range)
plt.title("Optimized GP analysis")

k_tools.plot_corrupted_samples(function_range, 
                               functions, 
                               sample_range, 
                               corrupted_samples,
                               plot_samples = "no")
GP_tools.plot_posterior_pdf(meta_results["optimized"]["pdf"], 
                            value_range, 
                            target_range)
plt.title("Optimized GP posterior densities")

## Marginalized GP analysis ##
k_tools.plot_corrupted_samples(function_range, 
                               functions, 
                               sample_range, 
                               corrupted_samples)
GP_tools.plot_GP_posterior(GP_posterior_expectation = meta_results["marginalized"]["expectation"], 
                           GP_posterior_variance = meta_results["marginalized"]["variance"],
                           target_range = target_range)
plt.title("Marginalized GP analysis")

GP_tools.plot_posterior_pdf(meta_results["marginalized"]["pdf"], 
                            value_range, 
                            target_range)
k_tools.plot_corrupted_samples(function_range, 
                               functions, 
                               sample_range, 
                               corrupted_samples,
                               plot_samples = "no")
plt.title("Marginalized GP posterior densities")



