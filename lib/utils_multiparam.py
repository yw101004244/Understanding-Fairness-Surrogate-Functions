import numpy as np
from scipy.optimize import minimize # for loss func minimization
from multiprocessing import Pool, Process, Queue
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt # for plotting stuff
import sys

def train_model(x, y, x_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sensitive_attrs, threshold, gamma=None):

    max_iter = 999999 # maximum number of iterations
    w0 = np.random.rand(x.shape[1]+2,)
    w0[-2] = 2 # height
    w0[-1] = 1 # width

    if apply_fairness_constraints == 0:
        constraints = []
    else:
        constraints = get_constraint_list_cov(x, y, x_control, sensitive_attrs, threshold)      


    if apply_accuracy_constraint == 0:
        w = minimize(fun = loss_function,
            x0 = w0,
            args = (x, y),
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = constraints
            )

    else: 

        # Train an unconstrained model and use its result as the start point of the constrained problem
        w = minimize(fun = loss_function,
            x0 = w0,
            args = (x, y),
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = []
            )

        old_w = deepcopy(w.x)
        

        def constraint_gamma_all(w, x, y,  initial_loss_arr):
            new_loss = loss_function(old_w, x, y)
            old_loss = sum(initial_loss_arr)
            return ((1.0 + gamma) * old_loss) - new_loss
        unconstrained_loss_arr = loss_function(old_w, x, y, return_arr=True)


        constraints = []
        c = ({'type': 'ineq', 'fun': constraint_gamma_all, 'args':(x,y,unconstrained_loss_arr)})
        constraints.append(c)

        def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
            weight_vec_temp = weight_vec[:-2]
            height = weight_vec[-2]
            width = weight_vec[-1]

            d_theta = np.dot(weight_vec_temp, x_in.T)
            phi_d_theta = phi(d_theta, height, width)            # mapping the distance 
            
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


def sigmoid(x):
    return 1/(1+np.exp((-1)*x))

def phi(x,width=20,height=1):
    # return height*(sigmoid(width*x) - 1/2)
    # return 2*sigmoid(x)-1
    # return 2*sigmoid(width*x)-1
    return x
    # return sigmoid(x)

def logistic_loss(w_all, X, y, return_arr=None):

    w = w_all[:-2]

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

def test_sensitive_attr_constraint_cov(model_all, x_arr, y_arr_dist_boundary, x_control, thresh):

    assert(x_arr.shape[0] == x_control.shape[0])
    if len(x_control.shape) > 1: # make sure we just have one column in the array
        assert(x_control.shape[1] == 1)
    
    arr = []
    if model_all is None:
        arr = y_arr_dist_boundary # simply the output labels
    else:
        model = model_all[:-2]
        arr = np.dot(model, x_arr.T) 


        arr = np.array(arr, dtype=np.float64)

        # mapping 
        height = model_all[-2]
        width = model_all[-1]
        arr = phi(arr, height, width) 


    arr = np.array(arr, dtype=np.float64)

    cov = np.dot(x_control - np.mean(x_control), arr) / float(len(x_control))

    ans = thresh - abs(cov) # will be <0 if the condition is not satisfied
    
    return ans



def get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs, threshold):
    
    constraints = []

    for attr in sensitive_attrs:

        attr_arr = x_control_train[attr]
        attr_arr = attr_arr.astype('int64')
        thresh = threshold[attr]


        c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(x_train, y_train, attr_arr, thresh)})
        
        constraints.append(c)
    
    return constraints







































































"""Don't need to read or change those things below"""
"""If you want to understand its principle, please refer to utils.py in the folder fair_classification"""

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

def print_covariance_sensitive_attrs(model, x_arr, y_arr_dist_boundary, x_control, sensitive_attrs):

    arr = []
    if model is None:
        arr = y_arr_dist_boundary # simplt the output labels
    else:
        arr = np.dot(model, x_arr.T) # the product with the weight vector -- the sign of this is the output label
    
    sensitive_attrs_to_cov_original = {}
    for attr in sensitive_attrs:
        attr_arr = x_control[attr]
        attr_arr.astype(int)
        bin_attr = check_binary(attr_arr) # check if the attribute is binary (0/1), or has more than 2 vals
        
        if bin_attr == False: # if its a non-binary sensitive feature, then perform one-hot-encoding
            attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)

        thresh = 0

        if bin_attr:
            cov = thresh - test_sensitive_attr_constraint_cov(None, x_arr, arr, np.array(attr_arr), thresh)
            sensitive_attrs_to_cov_original[attr] = cov
        else: # sensitive feature has more than 2 categorical values            
            
            cov_arr = []
            sensitive_attrs_to_cov_original[attr] = {}
            for attr_val, ind in index_dict.items():
                t = attr_arr_transformed[:,ind]
                cov = thresh - test_sensitive_attr_constraint_cov(None, x_arr, arr, t, thresh)
                sensitive_attrs_to_cov_original[attr][attr_val] = cov
                cov_arr.append(abs(cov))

            cov = max(cov_arr)
            
    return sensitive_attrs_to_cov_original

def get_correlations(model, x_test, y_predicted, x_control_test, sensitive_attrs):

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

def print_classifier_fairness_stats(acc_arr, correlation_dict_arr, cov_dict_arr, s_attr_name):
    
    correlation_dict = get_avg_correlation_dict(correlation_dict_arr)

    if 1 in correlation_dict[s_attr_name][1].keys():
        non_prot_pos = correlation_dict[s_attr_name][1][1]
    else:
        non_prot_pos = 0

    if 1 in correlation_dict[s_attr_name][0].keys():
        prot_pos = correlation_dict[s_attr_name][0][1]
    else:
        prot_pos = 0

    # p_rule = min(prot_pos / non_prot_pos, non_prot_pos / prot_pos) * 100.0
    violation_of_disparate_impact = abs(prot_pos - non_prot_pos)
    
    print ("Accuracy: {:.4f}%".format(np.mean(acc_arr)*100))
    print ("Protected/non-protected in +ve class: {:.3f}% / {:.3f}%".format(prot_pos, non_prot_pos))
    # print ("P-rule achieved: {:.3f}%".format(p_rule))
    print ("violation_of_disparate_impact achieved: {:.3f}%".format(violation_of_disparate_impact))
    print ("Covariance between sensitive feature and decision from distance boundary : %0.3f" % (np.mean([v[s_attr_name] for v in cov_dict_arr])))
    print
    return violation_of_disparate_impact



def add_intercept(x):

    """ Add intercept to the data before linear classification """
    m,n = x.shape
    intercept = np.ones(m).reshape(m, 1) # the constant b
    return np.concatenate((intercept, x), axis = 1)

def check_accuracy(model, x_train, y_train, x_test, y_test):

    y_test_predicted = np.sign(np.dot(x_test, model))
    y_train_predicted = np.sign(np.dot(x_train, model))

    def get_accuracy(y, Y_predicted):
        correct_answers = (Y_predicted == y).astype(int) # will have 1 when the prediction and the actual label match
        accuracy = float(sum(correct_answers)) / float(len(correct_answers))
        return accuracy, sum(correct_answers)

    train_score, correct_answers_train = get_accuracy(y_train, y_train_predicted)
    test_score, correct_answers_test = get_accuracy(y_test, y_test_predicted)

    return train_score, test_score, correct_answers_train, correct_answers_test

def compute_p_rule(x_control, class_labels):
    non_prot_all = sum(x_control == 1.0) # non-protected group
    prot_all = sum(x_control == 0.0) # protected group
    non_prot_pos = sum(class_labels[x_control == 1.0] == 1.0) # non_protected in positive class
    prot_pos = sum(class_labels[x_control == 0.0] == 1.0) # protected in positive class
    frac_non_prot_pos = float(non_prot_pos) / float(non_prot_all)
    frac_prot_pos = float(prot_pos) / float(prot_all)
    p_rule = min(frac_prot_pos / frac_non_prot_pos,  frac_non_prot_pos / frac_prot_pos) * 100.0
    print
    print ("Total data points: %d" % (len(x_control)))
    print ("# non-protected examples: %d" % (non_prot_all))
    print ("# protected examples: %d" % (prot_all))
    print ("Non-protected in positive class: %d (%0.0f%%)" % (non_prot_pos, non_prot_pos * 100.0 / non_prot_all))
    print ("Protected in positive class: %d (%0.0f%%)" % (prot_pos, prot_pos * 100.0 / prot_all))
    print ("P-rule is: %0.0f%%" % ( p_rule ))
    return p_rule

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

    return x_all_train, y_all_train, x_control_all_train, x_all_test, y_all_test, x_control_all_test

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
    plt.xlabel("Predicted Label")
    plt.ylabel("Signed Distance")
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


    labels = ["$+1$(u)","$-1$(u)","$+1$(c)","$-1$(c)"]


    rt = plt.boxplot([d_theta_u_positive,d_theta_u_negative,d_theta_c_positive,d_theta_c_negative],
                      labels = labels,
                    #   notch = True,
                    #   vert = False,
                      sym = 'o',
                    #   whis = 3.0,
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
    # print("outlier size: ", outlier_size)
    # print("test set size: ", x_test.shape[0])


    # 0 means the line x=0 is drawed. 
    # 0.75 and 3.75 specify the location of the dashed line
    plt.hlines(0, 0.65, 4.35, colors = "r", linestyles = "dashed") 
    plt.xlabel(r'\textbf{Predicted Label} $\hat{y}$')
    plt.ylabel(r'\textbf{Signed Distance} $d_{\theta}(x)$')

    
    # plt.figure(dpi=100)
    plt.savefig('./boxplot.eps')
    plt.show()