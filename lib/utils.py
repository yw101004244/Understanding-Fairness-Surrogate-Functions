import numpy as np
from random import seed, shuffle
import loss_funcs as lf # our implementation of loss funcs
from scipy.optimize import minimize # for loss func minimization
from multiprocessing import Pool, Process, Queue
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt # for plotting stuff
import sys

# SEED = 1122334455
# seed(SEED) # set the random seed so that the random permutations can be reproduced again
# np.random.seed(SEED)

# plt.rcParams['text.usetex'] = True


def train_model(x, y, x_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma=None):

    """

    Function that trains the model subject to various fairness constraints.
    If no constraints are given, then simply trains an unaltered classifier.
    Example usage in: "synthetic_data_demo/decision_boundary_demo.py"

    ----

    Inputs:

    X: (n) x (d+1) numpy array -- n = number of examples, d = number of features, one feature is the intercept
    y: 1-d numpy array (n entries)
    x_control: dictionary of the type {"s": [...]}, key "s" is the sensitive feature name, and the value is a 1-d list with n elements holding the sensitive feature values
    loss_function: the loss function that we want to optimize -- for now we have implementation of logistic loss, but other functions like hinge loss can also be added
    apply_fairness_constraints: optimize accuracy subject to fairness constraint (0/1 values)
    apply_accuracy_constraint: optimize fairness subject to accuracy constraint (0/1 values)
    sep_constraint: apply the fine grained accuracy constraint
        for details, see Section 3.3 of arxiv.org/abs/1507.05259v3
        For examples on how to apply these constraints, see "synthetic_data_demo/decision_boundary_demo.py"
    Note: both apply_fairness_constraints and apply_accuracy_constraint cannot be 1 at the same time
    sensitive_attrs: ["s1", "s2", ...], list of sensitive features for which to apply fairness constraint, all of these sensitive features should have a corresponding array in x_control
    sensitive_attrs_to_cov_thresh: the covariance threshold that the classifier should achieve (this is only needed when apply_fairness_constraints=1, not needed for the other two constraints)
    gamma: controls the loss in accuracy we are willing to incur when using apply_accuracy_constraint and sep_constraint

    ----

    Outputs:

    w: the learned weight vector for the classifier

    """


    assert((apply_accuracy_constraint == 1 and apply_fairness_constraints == 1) == False) # both constraints cannot be applied at the same time

    max_iter = 100000 # maximum number of iterations for the minimization algorithm

    if apply_fairness_constraints == 0:
        constraints = []
    else:
        constraints = get_constraint_list_cov(x, y, x_control, sensitive_attrs, sensitive_attrs_to_cov_thresh)      

    if apply_accuracy_constraint == 0: #its not the reverse problem, just train w with cross cov constraints

        f_args=(x, y)
        w = minimize(fun = loss_function,
            x0 = np.random.rand(x.shape[1],),
            args = f_args,
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = constraints
            )

    else:

        # train on just the loss function
        w = minimize(fun = loss_function,
            x0 = np.random.rand(x.shape[1],),
            args = (x, y),
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = []
            )

        old_w = deepcopy(w.x)
        

        def constraint_gamma_all(w, x, y,  initial_loss_arr):
            
            gamma_arr = np.ones_like(y) * gamma # set gamma for everyone
            new_loss = loss_function(w, x, y)
            old_loss = sum(initial_loss_arr)
            return ((1.0 + gamma) * old_loss) - new_loss

        def constraint_protected_people(w,x,y): # dont confuse the protected here with the sensitive feature protected/non-protected values -- protected here means that these points should not be misclassified to negative class
            return np.dot(w, x.T) # if this is positive, the constraint is satisfied
        def constraint_unprotected_people(w,ind,old_loss,x,y):
            
            new_loss = loss_function(w, np.array([x]), np.array(y))
            return ((1.0 + gamma) * old_loss) - new_loss

        constraints = []
        predicted_labels = np.sign(np.dot(w.x, x.T))
        unconstrained_loss_arr = loss_function(w.x, x, y, return_arr=True)

        if sep_constraint == True: # separate gamma for different people
            for i in range(0, len(predicted_labels)):
                if predicted_labels[i] == 1.0 and x_control[sensitive_attrs[0]][i] == 1.0: # for now we are assuming just one sensitive attr for reverse constraint, later, extend the code to take into account multiple sensitive attrs
                    c = ({'type': 'ineq', 'fun': constraint_protected_people, 'args':(x[i], y[i])}) # this constraint makes sure that these people stay in the positive class even in the modified classifier             
                    constraints.append(c)
                else:
                    c = ({'type': 'ineq', 'fun': constraint_unprotected_people, 'args':(i, unconstrained_loss_arr[i], x[i], y[i])})                
                    constraints.append(c)
        else: # same gamma for everyone
            c = ({'type': 'ineq', 'fun': constraint_gamma_all, 'args':(x,y,unconstrained_loss_arr)})
            constraints.append(c)

        def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
            
            d_theta = np.dot(weight_vec, x_in.T)
            phi_d_theta = phi(d_theta)            # mapping the distance 
            
            cross_cov = (x_control_in_arr - np.mean(x_control_in_arr)) * phi_d_theta
            return float(abs(sum(cross_cov))) / float(x_in.shape[0])


        w = minimize(fun = cross_cov_abs_optm_func,
            x0 = old_w,
            args = (x, x_control[sensitive_attrs[0]]),
            method = 'SLSQP',
            options = {"maxiter":100000},
            constraints = constraints
            )

    try:
        assert(w.success == True)
    except:
        print("Optimization problem did not converge.. Check the solution returned by the optimizer.")
        print("Returned solution is:")
        print(w)

    return w.x


def learnable_train_model(x, y, x_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sensitive_attrs, threshold, gamma=None):
    
    max_iter = 999999 # maximum number of iterations

    # initiate w
    w0 = np.random.rand(x.shape[1]+1,)
    # w0[-2] = 2 # height
    w0[-1] = 50 # width

    
    # Train an unconstrained model and use its result as the start point of the constrained problem
    w = minimize(fun = loss_function,
        x0 = w0[:-1],
        args = (x, y),
        method = 'SLSQP',
        options = {"maxiter":max_iter},
        constraints = []
        )

    


    # old_w还是没有最后一个维度的状态
    old_w = deepcopy(w.x)
    old_w = np.append(old_w, w0[-1])


    def constraint_gamma_all(w_all, x, y, initial_loss_arr):
        w = w_all[:-1]
        new_loss = loss_function(w, x, y)
        old_loss = sum(initial_loss_arr)
        return ((1.0 + gamma) * old_loss) - new_loss

    
    unconstrained_loss_arr = loss_function(old_w[:-1], x, y, return_arr=True)
    constraints = []
    c = ({
        'type': 'ineq', 
        'fun': constraint_gamma_all,
        'args':(x, y, unconstrained_loss_arr)
    })
    constraints.append(c)

    def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
        weight_vec_temp = weight_vec[:-1]
        # height = weight_vec[-2]
        width = weight_vec[-1]

        d_theta = np.dot(weight_vec_temp, x_in.T)
        phi_d_theta = phi(d_theta, width, height=None)            # mapping the distance 
        
        cross_cov = (x_control_in_arr - np.mean(x_control_in_arr)) * phi_d_theta
        return float(abs(sum(cross_cov))) / float(x_in.shape[0])


    w = minimize(fun = cross_cov_abs_optm_func,
        x0 = old_w,
        args = (x, x_control[sensitive_attrs[0]]),
        method = 'SLSQP',
        options = {"maxiter":100000},
        constraints = constraints
        )


    try:
        assert(w.success == True)
    except:
        print("Optimization problem did not converge.. Check the solution returned by the optimizer.")
        print("Returned solution is:")
        print(w)

    return w.x




def learnable_test_sensitive_attr_constraint_cov(model_all, x_arr, y_arr_dist_boundary, x_control, thresh):

    assert(x_arr.shape[0] == x_control.shape[0])
    if len(x_control.shape) > 1: # make sure we just have one column in the array
        assert(x_control.shape[1] == 1)
    
    arr = []
    if model_all is None:
        arr = y_arr_dist_boundary # simply the output labels
    else:
        model = model_all[:-1]
        arr = np.dot(model, x_arr.T) 


        arr = np.array(arr, dtype=np.float64)

        # mapping 
        # height = model_all[-2]
        width = model_all[-1]
        arr = phi(arr, width, height=None) 


    arr = np.array(arr, dtype=np.float64)

    cov = np.dot(x_control - np.mean(x_control), arr) / float(len(x_control))

    ans = thresh - abs(cov) # will be <0 if the condition is not satisfied
    
    return ans


def learnable_get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs, threshold):
    
    constraints = []

    for attr in sensitive_attrs:

        attr_arr = x_control_train[attr]
        attr_arr = attr_arr.astype('int64')
        thresh = threshold[attr]


        c = ({'type': 'ineq', 'fun': learnable_test_sensitive_attr_constraint_cov, 'args':(x_train, y_train, attr_arr, thresh)})
        
        constraints.append(c)
    
    return constraints







def compute_cross_validation_error(x_all, y_all, x_control_all, num_folds, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh_arr, gamma=None):


    """
    Computes the cross validation error for the classifier subject to various fairness constraints
    This function is just a wrapper of "train_model(...)", all inputs (except for num_folds) are the same. See the specifications of train_model(...) for more info.

    Returns lists of train/test accuracy (with each list holding values for all folds), the fractions of various sensitive groups in positive class (for train and test sets), and covariance between sensitive feature and distance from decision boundary (again, for both train and test folds).
    """

    train_folds = []
    test_folds = []
    n_samples = len(y_all)
    train_fold_size = 0.7 # the rest of 0.3 is for testing

    # split the data into folds for cross-validation
    for i in range(0,num_folds):
        perm = list(range(0,n_samples)) # shuffle the data before creating each fold
        shuffle(perm)
        x_all_perm = x_all[perm]
        y_all_perm = y_all[perm]
        x_control_all_perm = {}
        for k in x_control_all.keys():
            x_control_all_perm[k] = np.array(x_control_all[k])[perm]


        x_all_train, y_all_train, x_control_all_train, x_all_test, y_all_test, x_control_all_test = split_into_train_test(x_all_perm, y_all_perm, x_control_all_perm, train_fold_size)

        train_folds.append([x_all_train, y_all_train, x_control_all_train])
        test_folds.append([x_all_test, y_all_test, x_control_all_test])

    def train_test_single_fold(train_data, test_data, fold_num, output_folds, sensitive_attrs_to_cov_thresh):

        x_train, y_train, x_control_train = train_data
        x_test, y_test, x_control_test = test_data

        w = train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
        train_score, test_score, correct_answers_train, correct_answers_test = check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
        
        distances_boundary_test = (np.dot(x_test, w)).tolist()
        all_class_labels_assigned_test = np.sign(distances_boundary_test)
        correlation_dict_test = get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
        cov_dict_test = print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)

        distances_boundary_train = (np.dot(x_train, w)).tolist()
        all_class_labels_assigned_train = np.sign(distances_boundary_train)
        correlation_dict_train = get_correlations(None, None, all_class_labels_assigned_train, x_control_train, sensitive_attrs)
        cov_dict_train = print_covariance_sensitive_attrs(None, x_train, distances_boundary_train, x_control_train, sensitive_attrs)

        output_folds.put([fold_num, test_score, train_score, correlation_dict_test, correlation_dict_train, cov_dict_test, cov_dict_train])


        return


    output_folds = Queue()
    processes = [Process(target=train_test_single_fold, args=(train_folds[x], test_folds[x], x, output_folds, sensitive_attrs_to_cov_thresh_arr[x])) for x in range(num_folds)]

    # Run processes
    for p in processes:
        p.start()


    # Get the reuslts
    results = [output_folds.get() for p in processes]
    for p in processes:
        p.join()
    
    
    test_acc_arr = []
    train_acc_arr = []
    correlation_dict_test_arr = []
    correlation_dict_train_arr = []
    cov_dict_test_arr = []
    cov_dict_train_arr = []

    results = sorted(results, key = lambda x : x[0]) # sort w.r.t fold num
    for res in results:
        fold_num, test_score, train_score, correlation_dict_test, correlation_dict_train, cov_dict_test, cov_dict_train = res

        test_acc_arr.append(test_score)
        train_acc_arr.append(train_score)
        correlation_dict_test_arr.append(correlation_dict_test)
        correlation_dict_train_arr.append(correlation_dict_train)
        cov_dict_test_arr.append(cov_dict_test)
        cov_dict_train_arr.append(cov_dict_train)

    
    return test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr



def print_classifier_fairness_stats(acc_arr, correlation_dict_arr, cov_dict_arr, s_attr_name):
    
    correlation_dict = get_avg_correlation_dict(correlation_dict_arr)
    
    if 1 in correlation_dict[s_attr_name][1].keys():
        non_prot_pos = correlation_dict[s_attr_name][1][1]
    else:
        non_prot_pos = 0.0001

    if 1 in correlation_dict[s_attr_name][0].keys():
        prot_pos = correlation_dict[s_attr_name][0][1]
    else:
        prot_pos = 0.0001


    # p_rule = min(prot_pos / non_prot_pos, non_prot_pos / prot_pos) * 100.0
    # violation_of_disparate_impact = abs(prot_pos - non_prot_pos)/100
    violation_of_disparate_impact = abs(prot_pos - non_prot_pos)/100
    
    print ("Accuracy: {:.4f}%".format(np.mean(acc_arr)*100))
    print ("Protected/non-protected in +ve class: {:.3f}% / {:.3f}%".format(prot_pos, non_prot_pos))
    print("Violation_of_Disparate_Impact: {:.4f}".format(violation_of_disparate_impact))



    # print ("P-rule achieved: {:.4f}%".format(p_rule))
    # print ("Covariance between sensitive feature and decision from distance boundary : %0.3f" % (np.mean([v[s_attr_name] for v in cov_dict_arr])))
    # print
    return violation_of_disparate_impact



def compute_violation_of_disparate_impact(x_control, class_labels):

    """ Compute the p-rule based on Doctrine of disparate impact """

    non_prot_all = sum(x_control == 1.0) # non-protected group
    prot_all = sum(x_control == 0.0) # protected group
    non_prot_pos = sum(class_labels[x_control == 1.0] == 1.0) # non_protected in positive class
    prot_pos = sum(class_labels[x_control == 0.0] == 1.0) # protected in positive class

    frac_non_prot_pos = float(non_prot_pos) / float(non_prot_all)
    frac_prot_pos = float(prot_pos) / float(prot_all)

    # p_rule = min(frac_prot_pos / frac_non_prot_pos,  frac_non_prot_pos / frac_prot_pos) * 100.0
    violation_of_disparate_impact = abs(frac_prot_pos - frac_non_prot_pos)

    # print
    # print ("Total data points: %d" % (len(x_control)))
    # print ("# non-protected examples: %d" % (non_prot_all))
    # print ("# protected examples: %d" % (prot_all))
    # print ("Non-protected in positive class: %d (%0.0f%%)" % (non_prot_pos, non_prot_pos * 100.0 / non_prot_all))
    # print ("Protected in positive class: %d (%0.0f%%)" % (prot_pos, prot_pos * 100.0 / prot_all))
    # # print ("P-rule is: %0.0f%%" % ( p_rule ))
    # print("Violation_of_Disparate_Impact: {:.4f}".format(violation_of_disparate_impact))
    return violation_of_disparate_impact




def add_intercept(x):

    """ Add intercept to the data before linear classification """
    m,n = x.shape
    intercept = np.ones(m).reshape(m, 1) # the constant b
    return np.concatenate((intercept, x), axis = 1)

def check_binary(arr):
    "give an array of values, see if the values are only 0 and 1"
    s = sorted(set(arr))
    if s[0] == 0 and s[1] == 1:
        return True
    else:
        return False

def get_one_hot_encoding(in_arr):
    """
        input: 1-D arr with int vals -- if not int vals, will raise an error
        output: m (ndarray): one-hot encoded matrix
                d (dict): also returns a dictionary original_val -> column in encoded matrix
    """

    for k in in_arr:
        if str(type(k)) != "<type 'numpy.float64'>" and type(k) != int and type(k) != np.int64:
            print (str(type(k)))
            print ("************* ERROR: Input arr does not have integer types")
            return None
        
    in_arr = np.array(in_arr, dtype=int)
    assert(len(in_arr.shape)==1) # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    
    index_dict = {} # value to the column number
    for i in range(0,len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[i]
        index_dict[val] = i

    out_arr = []    
    for i in range(0,len(in_arr)):
        tup = np.zeros(num_uniq_vals)
        val = in_arr[i]
        ind = index_dict[val]
        tup[ind] = 1 # set that value of tuple to 1
        out_arr.append(tup)

    return np.array(out_arr), index_dict

def check_accuracy(model, x_train, y_train, x_test, y_test, y_train_predicted, y_test_predicted):

    """
    returns the train/test accuracy of the model
    we either pass the model (w)
    else we pass y_predicted
    """
    if model is not None and y_test_predicted is not None:
        print ("Either the model (w) or the predicted labels should be None")
        raise Exception("Either the model (w) or the predicted labels should be None")

    if model is not None:
        y_test_predicted = np.sign(np.dot(x_test, model))
        y_train_predicted = np.sign(np.dot(x_train, model))

    def get_accuracy(y, Y_predicted):
        correct_answers = (Y_predicted == y).astype(int) # will have 1 when the prediction and the actual label match
        accuracy = float(sum(correct_answers)) / float(len(correct_answers))
        return accuracy, sum(correct_answers)

    train_score, correct_answers_train = get_accuracy(y_train, y_train_predicted)
    test_score, correct_answers_test = get_accuracy(y_test, y_test_predicted)

    return train_score, test_score, correct_answers_train, correct_answers_test












































def hinge(x):
    x = np.maximum(np.zeros_like(x), (1-x)) # hinge function

    return x


def sigmoid(x):
    # x > 0: 1 / 1+e^-x
    # x <= 0: e^x / 1+e^x
    # return np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))

    # tanh(x) = 2 sigmoid(2x) - 1 -> sigmoid(x) = 1/2 * (1 + tanh(x/2)) 
    return .5 * (1 + np.tanh(.5 * x))


def phi(x,width=1/2,height=1):
    # return height*(sigmoid(width*x) - 1/2)
    # return hinge(x)

    # return 2*sigmoid(width*x)-1
    return x
    










































from sklearn.metrics import confusion_matrix

def print_data_stats(sens_attr, class_labels):
    """Print a few numbers about the data: Total number of points, number of
    protected examples and unprotected examples, and number of protected points
    in positive class, and number of unprotected points in positive class.

    Parameters
    -----------
    sens_attr: numpy array
     The sensitive attribute of shape=(number_points,).
    class_labels: nunmp
        The class labels of shape=(number_points,).
    """
    non_prot_all = sum(sens_attr == 1.0)  # non-protected group
    prot_all = len(sens_attr) - non_prot_all  # protected group
    non_prot_pos = sum(class_labels[sens_attr == 1.0] == 1.0)  # non_protected in positive class
    prot_pos = sum(class_labels == 1.0) - non_prot_pos  # protected in positive class
    frac_non_prot_pos = float(non_prot_pos) / float(non_prot_all)
    frac_prot_pos = float(prot_pos) / float(prot_all)
    print
    print("Total data points: %d" % len(sens_attr))
    print("# non-protected examples: %d" % non_prot_all)
    print("# protected examples: %d" % prot_all)
    print("# non-protected examples in positive class: %d (%0.1f%%)" % (non_prot_pos, non_prot_pos * 100.0 / non_prot_all))
    print("# protected examples in positive class: %d (%0.1f%%)" % (prot_pos, prot_pos * 100.0 / prot_all))

def get_positive_rate(y_predicted, y_true):
    """Compute the positive rate for given predictions of the class label.

    Parameters
    ----------
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).
    y_true: numpy array
        The true class labels of shape=(number_points,).

    Returns
    ---------
    pr: float
        The positive rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    pr = (tp+fp) / (tp+fp+tn+fn)
    return pr

def get_true_positive_rate(y_predicted, y_true):
    """Compute the true positive rate for given predictions of the class label.

    Parameters
    ----------
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).
    y_true: numpy array
        The true class labels of shape=(number_points,).

    Returns
    ---------
    tpr: float
        The true positive rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    tpr = tp / (tp+fn)
    return tpr

def compute_fairness_measures(y_predicted, y_true, sens_attr):
    """Compute value of demographic parity and equality of opportunity for given predictions.

    Parameters
    ----------
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).
    y_true: numpy array
        The true class labels of shape=(number_points,).
    sens_attr: numpy array
        The sensitive labels of shape=(number_points,).

    Returns
    ----------
    DDP: float
        The difference of demographic parity.
    DEO: float
        The difference of equality of opportunity.
    """
    positive_rate_prot = get_positive_rate(y_predicted[sens_attr==-1], y_true[sens_attr==-1])
    positive_rate_unprot = get_positive_rate(y_predicted[sens_attr==1], y_true[sens_attr==1])
    true_positive_rate_prot = get_true_positive_rate(y_predicted[sens_attr==-1], y_true[sens_attr==-1])
    true_positive_rate_unprot = get_true_positive_rate(y_predicted[sens_attr==1], y_true[sens_attr==1])
    DDP = positive_rate_unprot - positive_rate_prot
    DEO = true_positive_rate_unprot - true_positive_rate_prot

    return DDP, DEO

def get_accuracy(y_true, y_predicted):
    """Compute the accuracy for given predicted class labels.

    Parameters
    ----------
    y_true: numpy array
        The true class labels of shape=(number_points,).
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).

    Returns
    ---------
    accuracy: float
        The accuracy of the predictions.
    """
    correct_answers = (y_predicted == y_true).astype(int)  # will have 1 when the prediction and the actual label match
    accuracy = float(sum(correct_answers)) / float(len(correct_answers))
    return accuracy

def print_clf_stats(model, x_train, x_test, y_train, y_test, s_train, s_test):
    train_acc = ut.get_accuracy(np.sign(model.predict(x_train)), y_train)
    test_acc = ut.get_accuracy(np.sign(model.predict(x_test)), y_test)
    test_DDP, test_DEO = ut.compute_fairness_measures(model.predict(x_test), y_test, s_test)
    train_DDP, train_DEO = ut.compute_fairness_measures(model.predict(x_train), y_train, s_train)

    print(10*'-'+"Train"+10*'-')
    print("Accuracy: %0.4f%%" % (train_acc * 100))
    print("DDP: %0.4f%%" % (train_DDP * 100), "DEO: %0.4f%%" % (train_DEO * 100))
    print(10*'-'+"Test"+10*'-')
    print("Accuracy: %0.4f%%" % (test_acc * 100))
    print("DDP: %0.4f%%" % (test_DDP * 100), "DEO: %0.4f%%" % (test_DEO * 100))
















def logistic_loss(w, X, y, return_arr=None):

    yz = y * np.dot(X,w)

    if return_arr == True:
        out = -(log_logistic(yz))
    else:
        out = -np.sum(log_logistic(yz))
    return out

def log_logistic(X):

    if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X) # same dimensions and data types

    idx = X>0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out

def test_sensitive_attr_constraint_cov(model, x_arr, y_arr_dist_boundary, x_control, thresh, verbose):

    
    """
    The covariance is computed b/w the sensitive attr val and the distance from the boundary
    If the model is None, we assume that the y_arr_dist_boundary contains the distance from the decision boundary
    If the model is not None, we just compute a dot product of model and x_arr
    for the case of SVM, we pass the distance from boundary because the intercept in internalized for the class
    and we have computed the distance using the project function

    this function will return -1 if the constraint specified by thresh parameter is not satisfied
    otherwise it will return +1
    if the return value is >=0, then the constraint is satisfied
    """


    assert(x_arr.shape[0] == x_control.shape[0])
    if len(x_control.shape) > 1: # make sure we just have one column in the array
        assert(x_control.shape[1] == 1)
    
    arr = []
    if model is None:
        arr = y_arr_dist_boundary # simply the output labels
    else:
        arr = np.dot(model, x_arr.T) # the product with the weight vector -- the sign of this is the output label
    
    arr = np.array(arr, dtype=np.float64)


    arr = phi(arr)


    cov = np.dot(x_control - np.mean(x_control), arr) / float(len(x_control))

        
    ans = thresh - abs(cov) # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    # ans = thresh - cov # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    if verbose is True:
        print ("Covariance is")
        print(cov)
        print ("Diff is:")
        print(ans)
        print
    return ans

def print_covariance_sensitive_attrs(model, x_arr, y_arr_dist_boundary, x_control, sensitive_attrs):


    """
    reutrns the covariance between sensitive features and distance from decision boundary
    """

    arr = []
    if model is None:
        arr = y_arr_dist_boundary # simplt the output labels
    else:
        arr = np.dot(model, x_arr.T) # the product with the weight vector -- the sign of this is the output label
    

    sensitive_attrs_to_cov_original = {}
    for attr in sensitive_attrs:

        attr_arr = x_control[attr]

        attr_arr = attr_arr.astype(int)

        bin_attr = check_binary(attr_arr) # check if the attribute is binary (0/1), or has more than 2 vals
        

        if bin_attr == False: # if its a non-binary sensitive feature, then perform one-hot-encoding
            attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)

        thresh = 0

        if bin_attr:
            cov = thresh - test_sensitive_attr_constraint_cov(None, x_arr, arr, np.array(attr_arr), thresh, False)
            sensitive_attrs_to_cov_original[attr] = cov
        else: # sensitive feature has more than 2 categorical values            
            
            cov_arr = []
            sensitive_attrs_to_cov_original[attr] = {}
            for attr_val, ind in index_dict.items():
                t = attr_arr_transformed[:,ind]
                cov = thresh - test_sensitive_attr_constraint_cov(None, x_arr, arr, t, thresh, False)
                sensitive_attrs_to_cov_original[attr][attr_val] = cov
                cov_arr.append(abs(cov))

            cov = max(cov_arr)
            
    return sensitive_attrs_to_cov_original


def get_correlations(model, x_test, y_predicted, x_control_test, sensitive_attrs):
    

    """
    returns the fraction in positive class for sensitive feature values
    """

    if model is not None:
        y_predicted = np.sign(np.dot(x_test, model))
        
    y_predicted = np.array(y_predicted)
    
    out_dict = {}
    for attr in sensitive_attrs:

        attr_val = []
        for v in x_control_test[attr]: attr_val.append(v)
        assert(len(attr_val) == len(y_predicted))


        total_per_val = defaultdict(int)
        attr_to_class_labels_dict = defaultdict(lambda: defaultdict(int))

        for i in range(0, len(y_predicted)):
            val = attr_val[i]
            label = y_predicted[i]

            # val = attr_val_int_mapping_dict_reversed[val] # change values from intgers to actual names
            total_per_val[val] += 1
            attr_to_class_labels_dict[val][label] += 1

        class_labels = set(y_predicted.tolist())

        local_dict_1 = {}
        for k1,v1 in attr_to_class_labels_dict.items():
            total_this_val = total_per_val[k1]

            local_dict_2 = {}
            for k2 in class_labels: # the order should be the same for printing
                v2 = v1[k2]

                f = float(v2) * 100.0 / float(total_this_val)


                local_dict_2[k2] = f
            local_dict_1[k1] = local_dict_2
        out_dict[attr] = local_dict_1

    return out_dict



def get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs, sensitive_attrs_to_cov_thresh):

    """
    get the list of constraints to be fed to the minimizer
    """

    constraints = []


    for attr in sensitive_attrs:

        attr_arr = x_control_train[attr]

        attr_arr = attr_arr.astype('int64')

        attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)
                
        if index_dict is None: # binary attribute
            thresh = sensitive_attrs_to_cov_thresh[attr]
            c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(x_train, y_train, attr_arr_transformed,thresh, False)})
            constraints.append(c)
        else: # otherwise, its a categorical attribute, so we need to set the cov thresh for each value separately


            for attr_val, ind in index_dict.items():
                attr_name = attr_val                
                thresh = sensitive_attrs_to_cov_thresh[attr][attr_name]
                
                t = attr_arr_transformed[:,ind]
                c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(x_train, y_train, t ,thresh, False)})
                constraints.append(c)

    return constraints


# the dataset has been shuffled in advance
def split_into_train_test(x_all, y_all, x_control_all, train_fold_size):

    split_point = int(round(float(x_all.shape[0]) * train_fold_size))
    x_all_train = x_all[:split_point]
    x_all_test = x_all[split_point:]
    y_all_train = y_all[:split_point]
    y_all_test = y_all[split_point:]
    x_control_all_train = {}
    x_control_all_test = {}
    for k in x_control_all.keys():
        x_control_all_train[k] = x_control_all[k][:split_point]
        x_control_all_test[k] = x_control_all[k][split_point:]

    return x_all_train, y_all_train, x_control_all_train, x_all_test, y_all_test, x_control_all_test,split_point


def get_avg_correlation_dict(correlation_dict_arr):
    # make the structure for the correlation dict
    correlation_dict_avg = {}
    # print correlation_dict_arr
    for k,v in correlation_dict_arr[0].items():
        correlation_dict_avg[k] = {}
        for feature_val, feature_dict in v.items():
            correlation_dict_avg[k][feature_val] = {}
            for class_label, frac_class in feature_dict.items():
                correlation_dict_avg[k][feature_val][class_label] = []

    # populate the correlation dict
    for correlation_dict in correlation_dict_arr:
        for k,v in correlation_dict.items():
            for feature_val, feature_dict in v.items():
                for class_label, frac_class in feature_dict.items():
                    correlation_dict_avg[k][feature_val][class_label].append(frac_class)

    # now take the averages
    for k,v in correlation_dict_avg.items():
        for feature_val, feature_dict in v.items():
            for class_label, frac_class_arr in feature_dict.items():
                correlation_dict_avg[k][feature_val][class_label] = np.mean(frac_class_arr)

    return correlation_dict_avg



def plot_cov_thresh_vs_acc_pos_ratio(x_all, y_all, x_control_all, num_folds, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs):


    # very the covariance threshold using a range of decreasing multiplicative factors and see the tradeoffs between accuracy and fairness
    it = 0.05
    cov_range = np.arange(1.0, 0.0-it, -it).tolist()
    if apply_accuracy_constraint == True:
        if sep_constraint == False:
            it = 0.1
            cov_range = np.arange(0.0, 1.0 + it, it).tolist()
        if sep_constraint == True:
            cov_range =  [0,1,5,10,20,50,100,500,1000]

    
    positive_class_label = 1 # positive class is +1
    train_acc = []
    test_acc = []
    positive_per_category = defaultdict(list) # for each category (male / female), the frac of positive

    # first get the original values of covariance in the unconstrained classifier -- these original values are not needed for reverse constraint    
    test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr = compute_cross_validation_error(x_all, y_all, x_control_all, num_folds, loss_function, 0, apply_accuracy_constraint, sep_constraint, sensitive_attrs, [{} for i in range(0,num_folds)], 0)

    for c in cov_range:
        print ("LOG: testing for multiplicative factor: %0.2f" % c)
        sensitive_attrs_to_cov_original_arr_multiplied = []
        for sensitive_attrs_to_cov_original in cov_dict_train_arr:
            sensitive_attrs_to_cov_thresh = deepcopy(sensitive_attrs_to_cov_original)
            for k in sensitive_attrs_to_cov_thresh.keys():
                v = sensitive_attrs_to_cov_thresh[k]
                if type(v) == type({}):
                    for k1 in v.keys():
                        v[k1] = v[k1] * c
                else:
                    sensitive_attrs_to_cov_thresh[k] = v * c
            sensitive_attrs_to_cov_original_arr_multiplied.append(sensitive_attrs_to_cov_thresh)


        test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr  = compute_cross_validation_error(x_all, y_all, x_control_all, num_folds, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_original_arr_multiplied, c)
        test_acc.append(np.mean(test_acc_arr))


        correlation_dict_train = get_avg_correlation_dict(correlation_dict_train_arr)
        correlation_dict_test = get_avg_correlation_dict(correlation_dict_test_arr)
        
        # just plot the correlations for the first sensitive attr, the plotting can be extended for the other values, but as a proof of concept, we will jsut show for one
        s = sensitive_attrs[0]    
        
        for k,v in correlation_dict_test[s].items():
            if v.get(positive_class_label) is None:
                positive_per_category[k].append(0.0)
            else:
                positive_per_category[k].append(v[positive_class_label])
    
    positive_per_category = dict(positive_per_category)
    
    p_rule_arr = (np.array(positive_per_category[0]) / np.array(positive_per_category[1])) * 100.0
    

    ax = plt.subplot(2,1,1)
    plt.plot(cov_range, positive_per_category[0], "-o" , color="green", label = "Protected")
    plt.plot(cov_range, positive_per_category[1], "-o", color="blue", label = "Non-protected")
    ax.set_xlim([min(cov_range), max(cov_range)])
    plt.xlabel('Multiplicative loss factor')
    plt.ylabel('Perc. in positive class')
    if apply_accuracy_constraint == False:
        plt.gca().invert_xaxis()
        plt.xlabel('Multiplicative covariance factor (c)')
    ax.legend()

    ax = plt.subplot(2,1,2)
    plt.scatter(p_rule_arr, test_acc, color="red")
    ax.set_xlim([min(p_rule_arr), max(max(p_rule_arr), 100)])
    plt.xlabel('P% rule')
    plt.ylabel('Accuracy')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.show()


def get_line_coordinates(w, x1, x2):
    y1 = (-w[0] - (w[1] * x1)) / w[2]
    y2 = (-w[0] - (w[1] * x2)) / w[2]    
    return y1,y2













''' Compute the term without the sensitive attribute removed '''
''' It serves as a preparation. The removed version is after this function '''
def compute_term(w, x_test, sensitive_index, violation_of_disparate_impact):

    d_theta = np.dot(w, x_test.T)
    # d_theta = phi(d_theta)
    D_theta = np.abs(d_theta)
    y_pred = np.sign(d_theta)

    ''' compute A,B,C,D and their indices '''
    SENSITIVE_INDEX = sensitive_index

    # print("A+B+C+D =",x_test.shape[0])
    A_C_INDEX = np.array(x_test[:,SENSITIVE_INDEX]==0).astype(int) # equals 1 if in A+C. equals 0, in B+D
    B_D_INDEX = np.array(x_test[:,SENSITIVE_INDEX]==1).astype(int)

    
    # values of y_pred are -1 and 1. note that it is not the same as x_test[:,SENSITIVE_INDEX], whose values are 
    # 0 and 1. Thus, we need to convert -1 to 0 to satisfy logic computations below.
    y_pred[y_pred==-1] = 0 
    A_B_INDEX = y_pred
    C_D_INDEX = np.logical_not(A_B_INDEX)

    A_B = np.count_nonzero(A_B_INDEX == 1)
    # print("A+B =",A_B)
    C_D = np.count_nonzero(A_B_INDEX == 0)
    # print("C+D =",C_D)
    A_C = np.sum(A_C_INDEX)
    # print("A+C =", A_C)
    B_D = np.sum(B_D_INDEX)
    # print("B+D =", B_D)

    # A and A_INDEX
    A_INDEX = np.logical_and(A_C_INDEX, A_B_INDEX)
    A = np.sum(A_INDEX)
    # print("A = ", A)

    # B and B_INDEX
    B = A_B - A
    B_INDEX = np.logical_and(A_B_INDEX, B_D_INDEX)
    # print("B =", B)
    # print("B =", np.sum(B_INDEX))

    # C and C_INDEX
    C = A_C - A
    C_INDEX = np.logical_and(A_C_INDEX, C_D_INDEX)
    # print("C =",C)
    # print("C =", np.sum(C_INDEX))

    # D and D_INDEX
    D = B_D - B
    D_INDEX = np.logical_and(C_D_INDEX, B_D_INDEX)
    # print("D =",D)
    # print("D =", np.sum(D_INDEX))


    ''' compute the term '''
    N = x_test.shape[0]
    # term_part1 = N * N * violation_of_disparate_impact / (2 * A_C * B_D)

    T_A = np.sum(D_theta[A_INDEX==1]) - A
    T_B = np.sum(D_theta[B_INDEX==1]) - B
    T_C = np.sum(D_theta[C_INDEX==1]) - C
    T_D = np.sum(D_theta[D_INDEX==1]) - D
    
    S_A = T_A + A
    S_B = T_B + B
    S_C = T_C + C
    S_D = T_D + D

    # print("T_A: ", T_A)
    # print("T_B: ", T_B)
    # print("S_C: ", S_C)
    # print("S_D: ", S_D)

    beta = (S_A-S_C)/(A+C)-(S_B-S_D)/(B+D)
    # print("beta: ", beta)
    term_part1 = (2 * A_C * B_D) / (N * N) * beta
    # print("term_1: ", term_part1)


    # term_part2 = (1/2) * abs( (T_A - T_C)/(A + C) - (T_B - T_D)/(B + D) )
    # term_part2 = (1/2) * ( (T_A - T_C)/(A + C) - (T_B - T_D)/(B + D) )
    # print((T_A - S_C)/(A + C))
    # print((T_B - S_D)/(B + D))
    term_part2 = (T_A - S_C)/(A + C) - (T_B - S_D)/(B + D)
    # print("term_2: ", term_part2)



    # return the bound
    # term = abs(term_part1) + abs(term_part2)

    # return term



''' Now we compute the term with the sensitive attribute removed '''
def compute_term_removed(d_theta, a_c_index, b_d_index, n, violation_of_disparate_impact):

    y_pred = np.sign(d_theta)
    d_theta = phi(d_theta)
    D_theta = np.abs(d_theta)
    

    ''' compute A,B,C,D and their indices '''

    # print("A+B+C+D =",n)
    A_C_INDEX = a_c_index # equals 1 if in A+C. equals 0, in B+D
    B_D_INDEX = b_d_index
    N = n

    
    # values of y_pred are -1 and 1. note that it is not the same as x_test[:,SENSITIVE_INDEX], whose values are 
    # 0 and 1. Thus, we need to convert -1 to 0 to satisfy logic computations below.
    y_pred[y_pred==-1] = 0 
    A_B_INDEX = y_pred
    C_D_INDEX = np.logical_not(A_B_INDEX)

    A_B = np.count_nonzero(A_B_INDEX == 1)
    # print("A+B =",A_B)
    C_D = np.count_nonzero(A_B_INDEX == 0)
    # print("C+D =",C_D)
    A_C = np.sum(A_C_INDEX)
    # print("A+C =", A_C)
    B_D = np.sum(B_D_INDEX)
    # print("B+D =", B_D)

    # A and A_INDEX
    A_INDEX = np.logical_and(A_C_INDEX, A_B_INDEX)
    A = np.sum(A_INDEX)
    # print("A = ", A)

    # B and B_INDEX
    B = A_B - A
    B_INDEX = np.logical_and(A_B_INDEX, B_D_INDEX)
    # print("B =", B)
    # print("B =", np.sum(B_INDEX))

    # C and C_INDEX
    C = A_C - A
    C_INDEX = np.logical_and(A_C_INDEX, C_D_INDEX)
    # print("C =",C)
    # print("C =", np.sum(C_INDEX))

    # D and D_INDEX
    D = B_D - B
    D_INDEX = np.logical_and(C_D_INDEX, B_D_INDEX)
    # print("D =",D)
    # print("D =", np.sum(D_INDEX))


    ''' compute the term '''
    term_part1 = N * N * violation_of_disparate_impact / (4 * A_C * B_D)
    # print("term_1: ", term_part1)

    T_A = np.sum(D_theta[A_INDEX==1]) - A
    T_B = np.sum(D_theta[B_INDEX==1]) - B
    T_C = np.sum(D_theta[C_INDEX==1]) - C
    T_D = np.sum(D_theta[D_INDEX==1]) - D
    term_part2 = (1/2) * abs( (T_A - T_C)/(A + C) - (T_B - T_D)/(B + D) )
    # print("term_2: ", term_part2)

    term = term_part1 + term_part2
    return term




''' before mapping and after mapping '''
def draw_boxplot(w, x_test):
    labels = ['y_pred=+1', 'y_pred=-1']
    d_theta = np.dot(w, x_test.T)

    ''' before mapping '''
    d_theta_positive = d_theta[np.where(d_theta >= 0)]
    d_theta_negative = d_theta[np.where(d_theta < 0)]

    rt = plt.boxplot([d_theta_positive,d_theta_negative],
                      labels = labels,
                      notch = True,
                    #   vert = False,
                      sym = '.',
                    #   whis = 3.0,
                    #   showmeans=True,
                    #   meanline=True,
                      patch_artist=True,
                      showfliers=True)

    # print(rt['fliers'])

    # plt.violinplot([d_theta_positive,d_theta_negative],
    #                 vert = False,
    #                 quantiles = [[0.25,0.5,0.75],[0.25,0.5,0.75]]
    #                 )

    # 0 means the line x=0 is drawed. 
    # 0.75 and 2.25 specify the location of the dashed line
    plt.hlines(0, 0.75, 2.25, colors = "r", linestyles = "dashed") 
    plt.xlabel("Predicted Label", fontsize=15, fontweight='bold')
    plt.ylabel("Signed Distance", fontsize=15, fontweight='bold')
    plt.show()


    ''' after mapping '''
    # d_theta = phi(d_theta)
    # d_theta_positive = d_theta[np.where(d_theta >= 0)]
    # d_theta_negative = d_theta[np.where(d_theta < 0)]
    # rt = plt.boxplot([d_theta_positive,d_theta_negative],
    #                   labels = labels,
    #                   notch = True,
    #                 #   vert = False,
    #                   sym = '.',
    #                 #   whis = 3.0,
    #                 #   showmeans=True,
    #                 #   meanline=True,
    #                   patch_artist=True,
    #                   showfliers=True) 
    # plt.hlines(0, 0.75, 2.25, colors = "r", linestyles = "dashed") 
    # plt.xlabel("Predicted Label")
    # plt.ylabel("Signed Distance")
    # plt.show()



def draw_box_plot_together(w_u,w_c,x_test):
    '''
    w_u: unconstrained parameter
    w_c: constrained parameter
    x_test: the test set
    '''

    d_theta_u = np.dot(w_u, x_test.T)
    d_theta_c = np.dot(w_c, x_test.T)

   
    d_theta_u_positive = d_theta_u[np.where(d_theta_u >= 0)]
    d_theta_u_negative = d_theta_u[np.where(d_theta_u < 0)]

    d_theta_c_positive = d_theta_c[np.where(d_theta_c >= 0)]
    d_theta_c_negative = d_theta_c[np.where(d_theta_c < 0)]


    labels = ["+1(u)","-1(u)","+1(c)","-1(c)"]


    rt = plt.boxplot([d_theta_u_positive,d_theta_u_negative,d_theta_c_positive,d_theta_c_negative],
                      labels = labels,
                    #   notch = True,
                    #   vert = False,
                      sym = 'o',
                    #   whis = 2.0,
                    #   showmeans=True,
                    #   meanline=True,
                      patch_artist=True,
                      showfliers=True) 
    

    # some setting 
    temp = 1
    for box in rt["boxes"]:
        box.set( color='#7570b3', linewidth=2) # frame
        temp = temp + 1
        if(temp <= 3):
            box.set(color='#8A2BE2')
        else:
            box.set(color='#00FA9A')

    # for median in rt['medians']:
    #     median.set(color='DarkBlue', linewidth=3)

    outlier_size = 0
    for flier in rt['fliers']:
        # flier.set(marker='o', color='r', alpha=0.5)
        # outliers = flier.get_xydata()
        outlier_size = outlier_size + flier.get_xydata().shape[0]
    print("outlier size: ", outlier_size)
    print("test set size: ", x_test.shape[0])


    # 0 means the line x=0 is drawed. 
    # 0.75 and 3.75 specify the location of the dashed line
    plt.hlines(0, 0.65, 4.35, colors = "r", linestyles = "dashed") 
    # plt.xlabel(r'\textbf{Predicted Label} $\hat{y}$')
    # plt.ylabel(r'\textbf{Signed Distance} $d_{\theta}(x)$')
    plt.xlabel('Predicted Label', fontsize=15, fontweight='bold')
    plt.ylabel('Signed Distance', fontsize=15, fontweight='bold')

    
    # plt.figure(dpi=100)
    # plt.savefig('./boxplot_adult_general_sigmoid.jpg')
    plt.show()