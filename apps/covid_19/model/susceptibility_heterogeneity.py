import scipy.special as special
import numpy as np
import matplotlib.pyplot as plt


def get_gamma(coeff_var: float):
    """
    Use function described in caption to Extended Data Fig 1 of Aguas et al pre-print to produce gamma distribution
    from coefficient of variation and independent variable.

    :param coeff_var:
    Independent variable
    :return: callable
    Function that provides the gamma distribution from the coefficient of variation
    """

    recip_cv_2 = coeff_var ** -2.

    def gamma_func(x_value: float):
        return x_value ** (recip_cv_2 - 1.) * \
               np.exp(-x_value * recip_cv_2) / \
               special.gamma(recip_cv_2) / \
               coeff_var ** (2. * recip_cv_2)

    return gamma_func


def produce_gomes_exfig1():
    """
    Produce figure equivalent to Extended Data Fig 1 of Aguas et al pre-print
    :return:
    """

    gamma_plot = plt.figure()
    axis = gamma_plot.add_subplot(1, 1, 1)
    for coeff in [0.5, 1., 2.]:
        gamma_func = get_gamma(coeff)
        x_values = np.linspace(0.1, 3., 50)
        y_values = [gamma_func(i) for i in x_values]
        axis.plot(x_values, y_values)
    return gamma_plot
