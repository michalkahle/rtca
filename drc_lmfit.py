import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize
# from scipy import signal
# from sklearn import metrics


def logistic_fc(x, bottom, top, inflection, slope):
    return ((bottom-top)/(1+((x/inflection)**slope))) + top

def error_fc(params, x, y, logistic_fc):
    return logistic_fc(x, **params) - y

class Drc():
    """docstring for Drc"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs


# top_min=None, top_start=None, top_max=None,
#         bottom_min=None, bottom_start=None, bottom_max=None,
#         inflection_min=None, inflection_start=None, inflection_max=None,
#         slope_min=None, slope_start=None, slope_max=None,

    def fit(self, x, y):
        self.x = x
        self.y = y
        pp = Parameters()
        pp.add('bottom', value=self.kwargs.get('bottom_start', np.min(y)), min=0.)
        pp.add('top', value=np.max(y), min=0.8, max=1.2)
        pp.add('inflection', value=np.mean(x), min=x.min(), max=x.max())
        pp.add('slope', value=1.0, min=15.0)
        pp

        # fitter = Minimizer(error_fc, pp, fcn_args=(x, y, logistic_fc), nan_policy="omit")
        # model = fitter.minimize(method="least_squares")
        model = minimize(error_fc, pp, args=(x, y, logistic_fc), method='least_squares')

        self.coef_ = {p: model.params[p].value for p in model.params}

        self.res_fc = lambda x: logistic_fc(x, **self.coef_)
        return self



    def plot(self):
        plt.plot(self.x, self.y, '.')
        grid = np.linspace(self.x.min(), self.x.max(), 100)
        plt.plot(grid, self.res_fc(grid))
        plt.draw()

curves = [
    {
        'x' : np.array([2.5200e-08, 2.5200e-08, 2.5200e-08, 7.5000e-08, 7.5000e-08,
            7.5000e-08, 2.1256e-07, 2.1256e-07, 2.1256e-07, 6.2500e-07,
            6.2500e-07, 6.2500e-07, 1.6251e-06, 1.6251e-06, 1.6251e-06,
            4.6251e-06, 4.6251e-06, 4.6251e-06]),
        'y' : np.array([1.10541216, 0.98457478, 1.05534471, 1.0260343 , 1.06998378,
            0.96835968, 1.0040424 , 1.0018167 , 1.04387628, 0.71612711,
            0.7343082 , 0.67752642, 0.19808611, 0.22435655, 0.21059667,
            0.05259525, 0.05163417, 0.06471088])
    },{
        'x' : np.array([2.500e-08, 2.500e-08, 2.500e-08, 7.500e-08, 7.500e-08, 7.500e-08,
            2.125e-07, 2.125e-07, 2.125e-07, 6.250e-07, 6.250e-07, 6.250e-07,
            1.625e-06, 1.625e-06, 1.625e-06, 4.625e-06, 4.625e-06, 4.625e-06]),
        'y' : np.array([1.01747186, 1.02583084, 1.0907764 , 1.1364057 , 0.99933006,
            1.05905916, 1.03447475, 1.04009689, 0.98540766, 1.17867535,
            0.90583212, 1.11287499, 1.02934762, 1.07260772, 1.03361375,
            1.06071045, 1.00143045, 1.06973276])
    },{
        'x' : np.array([2.500e-08, 2.500e-08, 2.500e-08, 7.500e-08, 7.500e-08, 7.500e-08,
            2.125e-07, 2.125e-07, 2.125e-07, 6.250e-07, 6.250e-07, 6.250e-07,
            1.625e-06, 1.625e-06, 1.625e-06, 4.625e-06, 4.625e-06, 4.625e-06]),
        'y' : np.array([0.93390644, 0.98980323, 0.96982281, 0.67101681, 0.60951277,
            0.75006632, 0.20158636, 0.20228153, 0.20761667, 0.089036  ,
            0.07477118, 0.09920341, 0.0552651 , 0.05398145, 0.05050354,
            0.03929482, 0.04185425, 0.04895192])
    }
]

if __name__ == "__main__":
    pass