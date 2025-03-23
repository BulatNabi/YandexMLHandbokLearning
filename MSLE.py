import numpy as np
def root_mean_squared_logarithmic_error(y_true, y_pred, a_min=1.):
    for i in range(0, len(y_true)):
        if (y_true[i] < 0):
            raise Exception("Y_true must be not negative")
    for i in range(0, len(y_pred)):
        if (y_pred[i] < a_min):
            y_pred[i] = max(y_pred[i], a_min)

    db = (np.log(y_pred) - np.log(y_true)) ** 2
    return db.mean() ** 0.5