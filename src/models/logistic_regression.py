import numpy as np
import pandas as pd
from .optimizers import SGD, Momentum, NAG, RMSProp, AdaDelta, Adam

class LogisticRegression:
    '''
    Реализация логистической регрессии с различными оптимизаторами и регуляризацией
    '''

    def __init__(self,
                optimizer='sgd',
                learning_rate=0.01,
                reg_type=None,
                reg_value=None,
                batch_size=40):
        
        self.learning_rate = learning_rate
        self.reg_type = reg_type
        self.reg_value = reg_value
        self.batch_size = batch_size

        if optimizer == 'SGD': self.optimizer = SGD
        elif optimizer == 'Momentum': self.optimizer = Momentum
        elif optimizer == 'NAG': self.optimizer = NAG
        elif optimizer == 'RMSProp': self.optimizer = RMSProp
        elif optimizer == 'AdaDelta': self.optimizer = AdaDelta
        elif optimizer == 'Adam': self.optimizer = Adam
        else: raise ValueError(f'Введен неподдерживающийся оптимизатор {optimizer}')

        self.weights = None
        self.bias = None
        self.history_loss = []

    @ staticmethod
    def _sigmoid(z):
        '''Сигмоида'''
        z = np.asarray(z, dtype=float)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _binary_cross_entropy(self, p_true, p_pred):
        '''Бинарная кросс-энтропия'''
        p_pred = np.clip(p_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(p_true[:, 1] * np.log(p_pred) + p_true[:, 0] * np.log(1 - p_pred))

        if self.reg_type == "L1": loss += self.reg_value * np.sum(np.abs(self.weights))
        elif self.reg_type == "L2": loss += 0.5*self.reg_value**2 * np.sum(self.weights**2)

        return loss
    
    def _calculating_gradients(self, probability, sample_Y):
        '''
        Вычисление градиентов
        '''
        dz = probability - sample_Y
        db = (1 / self.batch_size) * np.sum(dz)
    
    def fit(self, X, Y, epochs=1000, verbose=True):
        '''
        Обучение
        '''
        # инициализация весов и биаса
        w = np.full(X.shape[-1], 0.1, dtype='float')
        b = 0.1

        # процесс обучения
        for _ in range(epochs):
            # выборка для SGD
            rand_index = np.random.choice(X.shape[0], self.batch_size, replace=False)
            sample_X = X[rand_index]
            sample_Y = Y[rand_index]

            # подсчет вероятности
            linear_model = w @ sample_X.T + b
            probability = __class__._sigmoid(linear_model)

            # потери
            y_one_hot = np.eye(2)[sample_Y]
            loss = self._binary_cross_entropy(y_one_hot, probability)
            self.history_loss.append(loss)

            # подсчёт производных
            dz = probability - sample_Y
            dw = (1 / self.batch_size) * np.dot(sample_X.T, dz)
            db = (1 / self.batch_size) * np.sum(dz)



    


    