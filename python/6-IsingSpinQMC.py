get_ipython().magic('reload_ext pytriqs.magic')

from pytriqs.utility import mpi

get_ipython().run_cell_magic('triqs', '--only qmc', '#include <triqs/mc_tools/mc_generic.hpp>\n#include <triqs/utility/callbacks.hpp>\n\n// --------------- configuration : a spin +1, -1 ---------------\n\nstruct configuration {\n int spin = -1;\n};\n\n// --------------- a move: flip the spin ---------------\nstruct flip {\n configuration* config;\n double beta_h;\n\n flip(configuration* config_, double beta, double h) : config(config_), beta_h(beta * h) {}\n\n double attempt() { return std::exp(-2 * config->spin * beta_h); }\n\n double accept() {\n  config->spin *= -1;\n  return 1.0;\n }\n\n void reject() {}\n};\n\n//  ----------------- a measurement: the magnetization ------------\nclass compute_m {\n configuration const * config;\n double& avg_magn;\n double Z = 0, M = 0;\n public:   \n\n compute_m(configuration* config_, double& avg_magn) : config(config_), avg_magn(avg_magn) {}\n\n void accumulate(double sign) {\n  Z += sign;\n  M += sign * config->spin;\n }\n\n void collect_results(triqs::mpi::communicator c) {\n  avg_magn = M/Z;\n }\n};\n\n//  ----------------- main ------------\n\ndouble qmc(double beta, double field) { \n  \n triqs::mpi::communicator world;\n    \n // #parameters of the Monte Carlo\n int n_cycles = 5000000;\n int length_cycle = 10;\n int n_warmup_cycles = 10000;\n std::string random_name = "";\n int random_seed = 374982 + world.rank() * 273894;\n int verbosity = (world.rank() == 0 ? 2 : 0);\n\n // #Generic Monte Carlo\n triqs::mc_tools::mc_generic<double> SpinMC(random_name, random_seed, 1.0, verbosity);\n\n configuration config;\n double mag;\n\n // #add moves and measures\n SpinMC.add_move(flip(&config, beta, field), "flip move");\n SpinMC.add_measure(compute_m(&config, mag), "magnetization measure");\n\n // #Run and collect results\n SpinMC.warmup_and_accumulate(n_warmup_cycles, n_cycles, length_cycle, triqs::utility::clock_callback(600));\n SpinMC.collect_results(world);\n //std::cout << "Finished calculation for field = " << field << "." << std::endl;\n return mag;\n}')

import numpy as np
X = np.arange(0.1,2,0.2)
r = [qmc(2, h) for h in X]

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
r_theo = [(1- np.exp(-2 * 2*h))/ (1+ np.exp(-2 * 2*h)) for h in X]
plt.plot(X, r, '-o', label='Calculated')
plt.plot(X, r_theo, label='Theoretical')
plt.xlim(0,2)
plt.ylim(0,1)
plt.legend()



