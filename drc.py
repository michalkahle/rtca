import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import inf
# from sklearn import metrics

def hill(x, left, right, inflection, slope):
    return ((right-left)/(1+((x/inflection)**slope))) + left

class Drc():
    """docstring for Drc"""
    def __init__(self, start=None, lower_bounds=None, upper_bounds=None):
        self.start = start
        self.coef_ = None

    def fit(self, x, y):
        self.x = x
        self.y = y

        start = [y[0], y[-1], np.mean(x), 1.0]
        lbounds = [y.min(), y.min(), x.min(), 1.0]
        ubounds = [y.max(), y.max(), x.max(), inf]

        self.coef_, pcov = curve_fit(hill, x, y, p0=start, bounds=(lbounds, ubounds))
        return self

    def predict(self, x):
        return hill(x, *self.coef_)

    def plot(self):
        plt.plot(self.x, self.y, '.')
        if self.coef_ is not None:
            grid = np.linspace(self.x.min(), self.x.max(), 100)
            plt.plot(grid, self.predict(grid))
        plt.draw()

class Drct():
    """docstring for Drc"""
    def __init__(self, start=None, lower_bounds=None, upper_bounds=None):
        self.start = start
        self.coef_ = None

    def fit(self, x, Y):
        self.x = x
        self.Y = Y
        self.coef_ = np.zeros([Y.shape[0], 4], float)
        inflection, slope = 3, 4

        for i in range(Y.shape[0]-1, -1, -1):
            yi = Y[i]
            diff = np.abs((yi[:3] - yi[-3:]).sum() / 3)

            ubounds = [yi.max(), yi.max(), x.max(), 100]
            start = [yi[0], yi[-1], np.mean(x), 1.0]
            lbounds = [yi.min(), yi.min(), x.min(), 1.0]
            # if i < Y.shape[0]-1 and diff < 1:
            #     previous = self.coef_[i - 1]

            #     ubounds[inflection] = previous[inflection] + abs(diff)
            #     lbounds[inflection] = previous[inflection] - abs(diff)
            #     start[inflection] = previous[inflection]


            self.coef_[i], pcov = curve_fit(hill, x, yi, p0=start, bounds=(lbounds, ubounds))

        return self

    def predict(self, x):
        res = np.zeros([self.coef_.shape[0], x.shape[0]], float)
        for i in range(self.coef_.shape[0]):
            res[i] = hill(x, *self.coef_[i])
        return res

    def plot(self):
        plt.plot(self.x, self.y, '.')
        if self.coef_ is not None:
            grid = np.linspace(self.x.min(), self.x.max(), 100)
            plt.plot(grid, self.predict(grid))
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
    data = drc.curves[2]
    x = np.log10(data['x'])
    y = data['y']
    regr = drc.Drc()
    regr.fit(x, y)
    regr.plot()
    print(regr.coef_)