import numpy as np


class PerceptronSimple:

    def __init__(self, activation):
        """
        Inicializador de la clase.
        params:
        - activation: string con valores 'escalon', 'sigmoide', 'relu', 'leaky_relu', 'elu' o 'tanh'
        """
        # Inicialización de pesos aleatorios con valores entre -1 y 1
        self.weights = 2 * np.random.random((2, 1)) - 1
        self.activation = activation

    @staticmethod
    def escalon(x):
        """
        Función escalón. Devuelve 1 si x > 0.5 ó 0 en el resto de los casos
        """
        return 1 if x > 0.5 else 0

    @staticmethod
    def sigmoide(x):
        """
        Función sigmoide
        """
        return 1 / (1 - np.exp(-x)) if x != 0 else 1

    @staticmethod
    def derivadaSigmoide(x):
        """
        Derivada de la función sigmoide
        """
        return x * (1 - x)

    @staticmethod
    def relu(x):
        """
        Función ReLU - Rectified Linear Unit
        """
        return x if x > 0 else 0

    @staticmethod
    def derivadaRelu(x):
        """
        Derivada de la función ReLU
        """
        return 1 if x > 0 else 0

    @staticmethod
    def leakyRelu(x, alpha=0.01):
        """
        Función Leaky ReLU.
        """
        return x if x > 0 else alpha * x

    @staticmethod
    def derivadaLeakyRelu(x, alpha=0.01):
        """
        Derivada de la función Leaky ReLU
        """
        return 1 if x > 0 else alpha

    @staticmethod
    def elu(x, alpha=0.01):
        """
        Función ELU
        """
        return x if x > 0 else alpha * (np.exp(x) - 1)

    @staticmethod
    def derivadaElu(x, alpha=0.01):
        """
        Derivada de la función ELU
        """
        return 1 if x > 0 else alpha * np.exp(x)

    @staticmethod
    def tanh(x):
        """
        Función Tangente hiperbólica
        """
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def derivadaTanh(x):
        """
        Derivada de la función tangente hiperbólica
        """
        return 1 - np.tanh(x) ** 2
    
    def forwardPass(self, X):
        """
        Paso hacia adelante de información de la red
        params:
        - X: array unidimensional de features de un ejemplo
        
        NOTA: Hay que utilizar la activación correspondiente. Se accede a ella con self.activation
        
        Pseudocodigo:
        activation = self.activation
        if activation == "escalon":
            TO DO
        elif activation == "sigmoide":
            TO DO
        
        """
        activation = self.activation
        if activation == "escalon":
            return self.escalon(np.sum(self.weights * X[0]))
        elif activation == "sigmoide":
            return self.sigmoide(np.sum(self.weights * X[0]))
        elif activation == 'relu':
            return self.relu(np.sum(self.weights * X[0]))
        elif activation == 'leaky relu':
            return self.leakyRelu(np.sum(self.weights * X[0]))
        elif activation == 'elu':
            return self.elu(np.sum(self.weights * X[0]))
        elif activation == 'tanh':
            return self.tanh(np.sum(self.weights * X[0]))

    def fit(self, X_train, y_train, n_epochs=10, l_rate=0.1):
        """
        Entrenamiendo usando Descenso del Gradiente Estocástico
        Los pesos se actualizan con cada ejemplo que se pasa
        """
        activation = self.activation
        for epoch in range(n_epochs):
            sum_error = 0
            for x, y in zip(X_train, y_train):  # Para cada ejemplo de entrenamiento...
                output = self.forwardPass([x])
                sum_error += (y - output) ** 2
                if activation == "escalon":                    
                    # IMPLEMENTAR
                    # Primero comprobar si ha habido error
                    # Después actualizar los pesos igual que vimos en la teoría del perceptrón simple
                    if y == 1 and output < 0:
                        print(x[np.newaxis].T)
                        self.weights += x[np.newaxis].T
                    elif y == 0 and output > 0:
                        print(x[np.newaxis].T)
                        self.weights -= x[np.newaxis].T
                elif activation == "sigmoide":
                    # En este caso ya damos la actualización directamente
                    error = y - output
                    ajuste = x * np.array(error * self.derivadaSigmoide(output))
                    self.weights += l_rate * ajuste[np.newaxis].T
                elif activation == 'relu':
                    error = y - output
                    ajuste = x * np.array(error * self.derivadaRelu(output))
                    self.weights += l_rate * ajuste[np.newaxis].T
                elif activation == 'leaky relu':
                    error = y - output
                    ajuste = x * np.array(error * self.derivadaLeakyRelu(output))
                    self.weights += l_rate * ajuste[np.newaxis].T
                elif activation == 'elu':
                    error = y - output
                    ajuste = x * np.array(error * self.derivadaElu(output))
                    self.weights += l_rate * ajuste[np.newaxis].T
                elif activation == 'tanh':
                    error = y - output
                    ajuste = x * np.array(error * self.derivadaTanh(output))
                    self.weights += l_rate * ajuste[np.newaxis].T
            print("Epoch: " + str(epoch) + ", Error: " + str(sum_error) + ', Learning Rate: ' + str(l_rate))

    def predict(self, X_new):
        """
        Predicción de ejemplos nuevos
        Similar a forwardPass pero contemplando el caso de varios ejemplos a la vez
        """
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        for i, x in enumerate(X_new):
            activation = self.activation
            if activation == "escalon":
                y[i] = self.escalon(np.sum(self.weights * x))
            elif activation == "sigmoide":
                y[i] = self.sigmoide(np.sum(self.weights * x))
            elif activation == "relu":
                y[i] = self.relu(np.sum(self.weights * x))
            elif activation == "leaky relu":
                y[i] = self.leakyRelu(np.sum(self.weights * x))
            elif activation == "elu":
                y[i] = self.elu(np.sum(self.weights * x))
            elif activation == "tanh":
                y[i] = self.tanh(np.sum(self.weights * x))
        return y


if __name__ == "__main__":
    X_train = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    y_train = np.array([[0, 1, 1, 0]]).T
    X_train_v2 = np.array([[2.78, 2.55], [1.46, 2.36], [3.39, 4.40], [1.38, 1.85], [3.06, 3.00], [7.62, 2.75],
                           [5.33, 2.08], [6.92, 1.77], [8.67, -0.24], [7.67, 3.50]])
    y_train_v2 = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T
    n_epochs = 10
    l_rate = 0.1
    X_test = np.array([[1, 0, 0], [0, 0, 1]])
    X_test_v2 = np.array([[2.78, 2.55], [1.46, 2.36], [3.39, 4.40], [1.38, 1.85], [3.06, 3.00], [7.62, 2.75],
                          [5.33, 2.08], [6.92, 1.77], [8.67, -0.24], [7.67, 3.50]])
    y_test_v2 = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T

    perceptronEscalon = PerceptronSimple("escalon")
    print("Perc. Escalon inicializado con: ", perceptronEscalon.weights)
    perceptronEscalon.fit(X_train_v2, y_train_v2, n_epochs, l_rate)
    print("Perc. Escalon se ha optimizado a: ", perceptronEscalon.weights)
    pred_escalon = perceptronEscalon.predict(X_test_v2)
    print("Pred escalon: ", pred_escalon)

    perceptronSigmoidal = PerceptronSimple("sigmoide")
    print("Perc. Sigmoidal inicializado con: ", perceptronSigmoidal.weights)
    perceptronSigmoidal.fit(X_train_v2, y_train_v2, n_epochs, l_rate)
    print("Perc. Sigmoidal se ha optimizado a: ", perceptronSigmoidal.weights)
    pred_sigmoide = perceptronSigmoidal.predict(X_test_v2)
    print("Pred sigmoide: ", pred_sigmoide)

    perceptronRelu = PerceptronSimple("relu")
    print("Perc. ReLU inicializado con: ", perceptronRelu.weights)
    perceptronRelu.fit(X_train_v2, y_train_v2, n_epochs, l_rate)
    print("Perc. ReLU se ha optimizado a: ", perceptronRelu.weights)
    pred_relu = perceptronRelu.predict(X_test_v2)
    print("Pred ReLU: ", pred_relu)

    perceptronLeakyrelu = PerceptronSimple("leaky relu")
    print("Perc. Leaky ReLU inicializado con: ", perceptronLeakyrelu.weights)
    perceptronLeakyrelu.fit(X_train_v2, y_train_v2, n_epochs, l_rate)
    print("Perc. Leaky ReLU se ha optimizado a: ", perceptronLeakyrelu.weights)
    pred_leakyRelu = perceptronLeakyrelu.predict(X_test_v2)
    print("Pred Leaky ReLU: ", pred_leakyRelu)

    perceptronElu = PerceptronSimple("elu")
    print("Perc. ELU inicializado con: ", perceptronElu.weights)
    perceptronElu.fit(X_train_v2, y_train_v2, n_epochs, l_rate)
    print("Perc. ELU se ha optimizado a: ", perceptronElu.weights)
    pred_elu = perceptronElu.predict(X_test_v2)
    print("Pred ELU: ", pred_elu)

    perceptronTanh = PerceptronSimple("tanh")
    print("Perc. Tanh inicializado con: ", perceptronTanh.weights)
    perceptronTanh.fit(X_train_v2, y_train_v2, n_epochs, l_rate)
    print("Perc. Tanh se ha optimizado a: ", perceptronTanh.weights)
    pred_tanh = perceptronTanh.predict(X_test_v2)
    print("Pred Tanh: ", pred_tanh)
