import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.special import rel_entr
from math import log
from random import shuffle
import time
from sklearn.ensemble import RandomForestClassifier




# ---------- Attack Classes ------------ #
class Attack:
    """
    A high-level class to represent the main common features of all of the attacks
    """
    def __init__(self):
        pass
    
    
    def run_attack(self):
        """
        General function to run the membership inference attack
        The goal is to guess, for each group of attacked_groups, if target is in the group or not.
        The adversary has access to the aggregated information computed from the users in group and in a non-overlapping reference set of users
        Only the timeslots mentioned in timeslots_set are actually used in the attacks, the other ones are not taken into account.
        """
        print("Must be overridden by a Children class")
    
class Binary_classifier(Attack):
    """
    A class to train a binary classifier over a number of train_aggregates and then tests this classifier on a number of test_aggregates
    in order to detect membership of the target trace within aggregates. Aggregates may already be bucket suppressed with some threshold k.
    The associated test and training labels indicate membership (1) or not (0) of the target trace within the aggregate
    We consider that the classifier may be implemented with logistic regression (default), random forest, or multi-layer perceptron
    """

    def __init__(self, target, train_aggregates, train_labels, test_aggregates = None, test_labels = [], validation_aggregates = None, validation_labels = []):
        """
        Input:
            target: target_id of user who is being attacked by Adv in the MIA
            train_aggregates: aggregate matrices for training MIA (ndarray with flattened n_rois*n_epochs vectors as rows)
            train_labels: labels indicating whether target was in a training aggregate
            test_aggregates: aggregate matrices for testing MIA (ndarray with flattened n_rois*n_epochs vectors as rows)
            test_labels: labels indicating whether target was in a test aggregate
            OPTIONAL
            validation_aggregates: aggregate matrices for validating the trained classifier (ndarray with flattened n_rois*n_epochs vectors as rows)
            validation_labels: labels indicating whether target was in a validation aggregate
        """
        Attack.__init__(self)
        self.target = target
        # reformat lists of train and test aggregates as 2d np array for downstream usage by sklearn scaler and classifier
        self.train_aggregates = np.stack([x.toarray().flatten() for x in train_aggregates])
        if test_aggregates:
            self.test_aggregates = np.stack([x.toarray().flatten() for x in test_aggregates])
        if validation_aggregates:
            self.validation_aggregates = np.stack([x.toarray().flatten() for x in validation_aggregates])
        else:
            self.validation_aggregates = np.empty((0,))
        self.validation_labels = validation_labels
        self.train_labels = train_labels
        self.test_labels = test_labels
    
    def set_test_aggregate(self, aggregate, label):
        self.test_aggregates = np.stack([aggregate.toarray().flatten()])
        self.test_labels = [label]

    def train_model(self, classification, scaler_type, pca_components):
        """
        Scale the data, apply PCA (optional), and train the classifier to perform membership inference attacks 
        Input:
            pca_components: 0 is reserved for no pca (keep all features)
        Output:
            classifier: Logistic Regression classifier trained on the datasets generated from the prior knowledge
            pca: PCA components fitted on the datasets generated from the prior knowledge
            scaler: MinMaxScaler fitted on the datasets generated from the prior knowledge
        """
        # scale the data
        if scaler_type == 'Standard':
            scaler = StandardScaler() 
        elif scaler_type == 'MinMax':
            scaler = MinMaxScaler()
        if self.validation_aggregates.size>0:
            scaler.fit(np.concatenate((self.train_aggregates, self.validation_aggregates)))
        else:
            scaler.fit(self.train_aggregates)
        x_train_scaled = scaler.transform(self.train_aggregates)
        if self.validation_aggregates.size > 0:
            x_validation_scaled = scaler.transform(self.validation_aggregates)
        if pca_components > 0:
            # Perform PCA to keep only the interesting dimensions if pca_components is used (default is no, with pca_components=0)
            pca = PCA(n_components=pca_components)  
            pca.fit(x_train_scaled)
            x_train = pca.transform(x_train_scaled)
            if self.validation_aggregates.size > 0:
                pca.fit(x_validation_scaled)
                x_validation = pca.transform(x_validation_scaled)
            print(f'performed PCA on training and validation data')
        else:
            x_train = x_train_scaled
            if self.validation_aggregates.size > 0:
                x_validation = x_validation_scaled
            # return dummy PCA, which we will not be using
            pca = PCA(n_components=1)
        # shuffle the order before training
        train_groups_with_labels = list(zip(x_train, self.train_labels))
        shuffle(train_groups_with_labels)
        x_train, train_labels = zip(*train_groups_with_labels)
        if self.validation_aggregates.size > 0:
            val_groups_with_labels = list(zip(x_validation, self.validation_labels))
            shuffle(val_groups_with_labels)
            x_validation, validation_labels = zip(*val_groups_with_labels)
        # Train the Logistic Regression classifier
        if classification == 'LR':
            classifier = LogisticRegression(solver='liblinear', penalty='l1', tol=self.LR_tol, C=self.LR_c, max_iter = self.LR_max_iter, random_state=42) 
        elif classification == 'MLP':
            classifier = MLPClassifier(hidden_layer_sizes = (200, ), solver = 'sgd', random_state=42)
        elif classification == 'RF':
            classifier = RandomForestClassifier(n_estimators=self.n_trees, random_state=42, max_depth=self.max_depth)
        t1 = time.time()
        classifier.fit(x_train, train_labels)
        t2 = time.time()
        if self.validation_aggregates.size > 0:
            validation_probabilities = classifier.predict_proba(x_validation)[:, 1]
            fpr, tpr, thresholds = roc_curve(validation_labels, validation_probabilities)
            optimal_idx = np.argmax(tpr - fpr)
            validation_threshold = thresholds[optimal_idx]
            # Calculate validation accuracy
            validation_predictions = (validation_probabilities >= validation_threshold).astype(int)
            validation_accuracy = accuracy_score(validation_labels, validation_predictions)
            print(f"Validation Accuracy with threshold {validation_threshold}:", validation_accuracy)
            return classifier, pca, scaler, validation_threshold
        else:
            return classifier, pca, scaler, 0.5

    
    def set_RF_hyperparameters(self, n_trees, max_depth):
        # initializes hyperparameters for random forest classifier
        self.n_trees = n_trees
        self.max_depth = max_depth
    
    def set_LR_hyperparameters(self, LR_c, LR_max_iter, LR_tol):
        self.LR_c = LR_c
        self.LR_max_iter = LR_max_iter
        self.LR_tol = LR_tol
        
        
    def test_model(self, classifier, pca, scaler, pca_components, validation_threshold):
        """
        Input:
            classifier: classifier trained on the datasets generated from the dataset from which the released aggregates are generated
            pca: PCA object fitted on datasets generated from the location data from which the released aggregates are generated
            scaler: scaler object, fitted on the datasets generated from the location data from which the released aggregates are generated
            pca_components: number of pca components to be retained 
        Output:
            p_scores: the model's probabilities for membership of the target in each tested aggregate
            accuracy: overall accuracy of the attack (%) in identifying the target's membership status across each tested aggregate
            predictions: the raw predictions of the model for each tested aggregate
        """
        # Scale the data
        x_test_scaled = scaler.transform(self.test_aggregates)
        if pca_components > 0:
            # Perform the PCA to reduce the dimensionality
            x_test = pca.transform(x_test_scaled)
            print('PCA transformation applied on test aggregates')
        else:
            x_test = x_test_scaled
        # Test the classifier on the data and obtain its probabilities for the target's membership on each tested aggregate and the raw predictions for each tested aggregate (0 or 1)
        p_scores = classifier.predict_proba(x_test)[:, 1]
        predictions = (p_scores >= validation_threshold).astype('int')
        return p_scores, predictions, validation_threshold
    

    def run_attack(self, classification, scaler_type, pca_components):
        """
        Input:
            k: bucket suppression threshold
        Runs the baseline test MIA
        """
        classifier, pca, scaler, validation_threshold = self.train_model(classification, scaler_type, pca_components)
        return self.test_model(classifier, pca, scaler, pca_components, validation_threshold)

    
## Utility

def compute_acc_auc(y_true, y_pred, p_scores, sigm = False):
    """
    Compute the accuracy and the AUC of the ROC curve for the MIA's predicitions on a fixed target
    Input:
        y_true: true labels of the attacked groups
        y_pred: scores given as output of the classifier (all entries should be in [0,1])
        p_scores: probability scores of membership given by classifier
        OPTIONAL
        sigm: boolean determining if a sigmoid transformation should be applied to the p_scores to confine them in [0,1]
    Output:
        acc: Accuracy of the attack
        area: Area Under the ROC Curve
    """
    if sigm:
        p_scores = np.exp(p_scores)/(1+np.exp(p_scores))
    assert all(0<=y<=1 for y in p_scores), "the classifier scores should be within [0,1]"
    fpr, tpr, thresholds = roc_curve(y_true, p_scores)
    area = auc(fpr, tpr)
    acc = accuracy_score(y_true, y_pred)
    return acc, area