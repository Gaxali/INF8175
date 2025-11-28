import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        score = nn.as_scalar(self.run(x))
        return 1 if score >= 0 else -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):  # Processe un sample à la foi
                prediction = self.get_prediction(x)
                true_label = nn.as_scalar(y)
                
                if prediction != true_label:
                    converged = False
                    # Maj weights: w = w + y * x
                    self.w.update(x, true_label)


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        # Architecture: 1 -> 64 -> 32 -> 1
        self.w1 = nn.Parameter(1, 64)   # Entrée vers première couche cachée
        self.b1 = nn.Parameter(1, 64)   # Biais de la première couche cachée
        self.w2 = nn.Parameter(64, 32)  # Première vers deuxième couche cachée
        self.b2 = nn.Parameter(1, 32)   # Biais de la deuxième couche cachée
        self.w3 = nn.Parameter(32, 1)   # Deuxième couche cachée vers sortie
        self.b3 = nn.Parameter(1, 1)    # Biais de sortie

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        # Première couche cachée avec activation ReLU
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        # Deuxième couche cachée avec activation ReLU  
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h1, self.w2), self.b2))
        # Couche de sortie (activation linéaire pour la régression)
        output = nn.AddBias(nn.Linear(h2, self.w3), self.b3)
        return output

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        predictions = self.run(x) 
        return nn.SquareLoss(predictions, y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        learning_rate = 0.01 # Taux d'apprentissage initial
        batch_size = 50      # Taille du batch
        
        epoch = 0
        while True:
            total_loss = 0 
            batch_count = 0
            
            for x, y in dataset.iterate_once(batch_size): 
                loss = self.get_loss(x, y)
                loss_value = nn.as_scalar(loss)
                total_loss += loss_value
                batch_count += 1
                
                grads = nn.gradients(
                    loss, 
                    [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
                )
                
                self.w1.update(grads[0], -learning_rate)
                self.b1.update(grads[1], -learning_rate)
                self.w2.update(grads[2], -learning_rate)
                self.b2.update(grads[3], -learning_rate)
                self.w3.update(grads[4], -learning_rate)
                self.b3.update(grads[5], -learning_rate)
                
            epoch += 1
            
            if epoch > 5000:  # Limite le nombre d'itérations pour éviter les boucles infinies
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        self.w1 = nn.Parameter(784, 256)  # Entrée vers première couche cachée
        self.b1 = nn.Parameter(1, 256)    # Biais de la première couche cachée
        self.w2 = nn.Parameter(256, 128)  # Première vers deuxième couche cachée
        self.b2 = nn.Parameter(1, 128)    # Biais de la deuxième couche cachée
        self.w3 = nn.Parameter(128, 10)   # Deuxième couche cachée vers sortie
        self.b3 = nn.Parameter(1, 10)     # Biais de sortie

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        # Première couche cachée avec activation ReLU
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        # Deuxième couche cachée avec activation ReLU
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h1, self.w2), self.b2))
        # Couche de sortie (logits bruts, pas d'activation pour SoftmaxLoss)
        logits = nn.AddBias(nn.Linear(h2, self.w3), self.b3)
        return logits

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        logits = self.run(x)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        learning_rate = 0.1
        batch_size = 100
        precision_cible = 0.97  # 97% de précision cible
        
        for epoch in range(50):  # Maximum 50 époques
            perte_totale = 0
            nombre_batches = 0
            
            # Boucle d'entraînement
            for x, y in dataset.iterate_once(batch_size):
                perte = self.get_loss(x, y)
                perte_totale += nn.as_scalar(perte)
                nombre_batches += 1
                
                # Calcul des gradients et mise à jour des paramètres
                grads = nn.gradients(perte, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                
                self.w1.update(grads[0], -learning_rate)
                self.b1.update(grads[1], -learning_rate)
                self.w2.update(grads[2], -learning_rate)
                self.b2.update(grads[3], -learning_rate)
                self.w3.update(grads[4], -learning_rate)
                self.b3.update(grads[5], -learning_rate)
            
            # Calcul de la précision sur l'ensemble de validation
            precision = dataset.get_validation_accuracy()
            
            # Arrêt anticipé si la précision cible est atteinte
            if precision >= precision_cible:
                break
