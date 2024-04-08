import os,sys
lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'lib')
lib_path = os.path.normpath(lib_path)
sys.path.insert(0, lib_path)


import numpy as np
from prepare_bank_data import * 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import utils as ut
import utils_general_sigmoid as ut_g
import utils_balanced as ut_b
import utils_balanced_sensitive as ut_bs
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints

# compute_term = False # for unconstrained case, do not need to compute


apply_fairness_constraints = None
apply_accuracy_constraint = None
sep_constraint = None

loss_function = lf._logistic_loss
# loss_function = lf._negative_sigmoid

sensitive_attrs = ["age"]
sensitive_attrs_to_cov_thresh = {}
gamma = None

""" Split the data into train and test """
df, y, x_control = load_bank_additional_data()


# Other preparation which is connected to prepare_bank_data.py
# Because to compute the term, we cannot leave it in prepare_bank_data.py because
# the sensitive features will be removed
X = df.drop('duration',axis=1)
X = df.drop('age',axis=1)
X = X.drop('y',axis=1)
X = X.values
X = ut.add_intercept(X) # add intercept to X before applying the linear classifier


# X = X[:1000]
# y = y[:1000]
# x_control['age'] = x_control['age'][:1000]



def general_sigmoid_train_test_classifier():
    w = ut_g.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, width, gamma)
    train_score, test_score, correct_answers_train, correct_answers_test = ut_g.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
    distances_boundary_test = (np.dot(x_test, w)).tolist()
    all_class_labels_assigned_test = np.sign(distances_boundary_test)
    correlation_dict_test = ut_g.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
    cov_dict_test = ut_g.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs, width)
    violation_of_disparate_impact = ut_g.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])	
    
    # if compute_term == True:
    # 	term = ut.compute_term(w, x_test, SENSITIVE_INDEX, violation_of_disparate_impact) # for COMPAS, sensitive_index = 3 + 1(where 1 is the intercept)
        # print("Bound: ", term)
        
    # ut.draw_boxplot(w, x_test)
    ut.draw_box_plot_together(w_uncons, w, x_test)

    return w, violation_of_disparate_impact, test_score



# """ Model Selection for general sigmoid"""

# ''' 10-fold cross validation (for general sigmoid) '''
# N_SPLITS = 10 
# kf = KFold(n_splits=N_SPLITS)

# apply_fairness_constraints = 1 # set this flag to one since we want to optimize accuracy subject to fairness constraints
# apply_accuracy_constraint = 0
# sep_constraint = 0
# compute_term = True

# width_list = [1/8, 1/4, 1/2, 1, 2, 4, 8]
# vio_result_list = {}
# acc_result_list = {}
# select_w = [] # record result to select w 

# for width in width_list:

#     ''' inits and defs '''
#     vio_list = []
#     acc_list = []

#     ''' prepare the data '''
#     for train_index, test_index in kf.split(X):
#         x_train = X[train_index]
#         y_train = y[train_index]
#         x_test = X[test_index]
#         y_test = y[test_index]

#         x_control_train = {}
#         x_control_test = {}
        
#         for k in x_control.keys():
#             x_control_train[k] = x_control[k][train_index]
#             x_control_test[k] = x_control[k][test_index]


#         ''' train '''
#         for i in range(0,1):
#             temp = i/2000
#             sensitive_attrs_to_cov_thresh = {'age':temp}
#             # print(i)
#             # print("== Classifier with fairness constraint ==")
#             w_f_cons, p_f_cons, acc_f_cons  = general_sigmoid_train_test_classifier()

#             # if acc_f_cons > 0.65 and p_f_cons < 0.01:
#             vio_list.append(p_f_cons)
#             acc_list.append(acc_f_cons)

#     ''' after 10 times and we record the result for each selection of w'''
#     vio_result_list[width] = vio_list
#     acc_result_list[width] = acc_list
#     if len(vio_list) != 0:
#         # select_w.append(np.mean(vio_list) + np.std(vio_list))
#         select_w.append(np.mean(vio_list))
#     else:
#         select_w.append(np.inf)


# ''' after 10-fold training for every selection of w'''
# assert len(width_list)==len(vio_result_list), 'Bug to be solved'

# # print(select_w)
# w_index = select_w.index(min(select_w))
# w_final = width_list[w_index]
# print("finally choose w={:.2f}".format(w_final))


# acc_list = acc_result_list[w_final]
# vio_list = vio_result_list[w_final]
# acc_list = [float('{:.7f}'.format(i)) for i in acc_list]
# vio_list = [float('{:.7f}'.format(i)) for i in vio_list]
# method_list = ['General Sigmoid' for _ in range(len(acc_list))]

# print("Accuracy")
# print(acc_list)
# print("DDP")
# print(vio_list)
# print("Methods")
# print(method_list)
# print("\n")








# ''' after model selection, we now compare different methods'''

def train_test_classifier():
    w = ut.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
    train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
    distances_boundary_test = (np.dot(x_test, w)).tolist()
    all_class_labels_assigned_test = np.sign(distances_boundary_test)
    correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
    cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
    violation_of_disparate_impact = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])	
    

    return w, violation_of_disparate_impact, test_score


def balanced_sensitive_train_test_classifier():
    w = ut_bs.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, 1, A_C_INDEX, B_D_INDEX, gamma)
    train_score, test_score, correct_answers_train, correct_answers_test = ut_bs.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
    distances_boundary_test = (np.dot(x_test, w)).tolist()
    all_class_labels_assigned_test = np.sign(distances_boundary_test)
    correlation_dict_test = ut_bs.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
    cov_dict_test = ut_bs.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs, 1, A_C_INDEX, B_D_INDEX)
    violation_of_disparate_impact = ut_bs.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])	
    
    if compute_term == True:   
        # print("Round 1:")
        term_pre = 1 # for COMPAS, sensitive_index = 3 + 1(where 1 is the intercept)
        w = ut_bs.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, 1, A_C_INDEX, B_D_INDEX, gamma)
        d_theta = np.dot(w, x_test.T)
        term_now = ut_bs.compute_term_removed(d_theta, A_C_INDEX_TEST, B_D_INDEX_TEST, N_TEST)
        # print("t = ", term_now)
        
        # term_pre and term_now are results of two adjacent calculations
        while abs(term_pre - term_now) > 0.01:
            w = ut_bs.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, term_now, A_C_INDEX, B_D_INDEX, gamma)
            term_pre = term_now

            d_theta = np.dot(w, x_test.T)
            term_now = ut_bs.compute_term_removed(d_theta, A_C_INDEX_TEST, B_D_INDEX_TEST, N_TEST) # for COMPAS, sensitive_index = 3 + 1(where 1 is the intercept)
            term_now = term_now * 0.1 + term_pre* 0.9
            print("t = ", term_now)

        # show results
        print("Final:")
        print("t = ", term_now)
        train_score, test_score, correct_answers_train, correct_answers_test = ut_bs.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
        distances_boundary_test = (np.dot(x_test, w)).tolist()
        all_class_labels_assigned_test = np.sign(distances_boundary_test)
        correlation_dict_test = ut_bs.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
        cov_dict_test = ut_bs.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs, term_now, A_C_INDEX, B_D_INDEX)
        violation_of_disparate_impact = ut_bs.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])	
    
    # ut.draw_boxplot(w, x_test)

    return w, violation_of_disparate_impact, test_score


















''' begin training '''

vio_list_uncons = []
acc_list_uncons = []
method_list_uncons = []

vio_list_linear = []
acc_list_linear = []
method_list_linear = []

vio_list_balance = []
acc_list_balance = []
method_list_balance = []

vio_list_general_sigmoid = []
acc_list_general_sigmoid = []
method_list_general_sigmoid = []

vio_list_reduction = []
acc_list_reduction = []
method_list_reduction = []





""" 1. Classify the data while optimizing for accuracy """

""" Split the data into train and test (for linear) """
# train_fold_size = 0.7
# x_train, y_train, x_control_train, x_test, y_test, x_control_test,split_point = ut.split_into_train_test(X, y, x_control, train_fold_size)
N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=12345)
# kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=666)
for train_index, test_index in kf.split(X):
    x_train = X[train_index]
    y_train = y[train_index]
    x_test = X[test_index]
    y_test = y[test_index]

    x_control_train = {}
    x_control_test = {}
    
    for k in x_control.keys():
        x_control_train[k] = x_control[k][train_index]
        x_control_test[k] = x_control[k][test_index]


    
    ''' 1. Unconstrained '''
    
    print("\n")
    print("== Unconstrained (original) classifier ==")

    apply_fairness_constraints = 0
    apply_accuracy_constraint = 0
    sep_constraint = 0
    compute_term = True

    w_uncons, p_uncons, acc_uncons = train_test_classifier()
    vio_list_uncons.append(p_uncons)
    acc_list_uncons.append(acc_uncons)
    method_list_uncons.append('Unconstrained')
    
   
    





    ''' 2. Balanced  Surrogate (based on sensitive feature)'''
    apply_fairness_constraints = 1 # set this flag to one since we want to optimize accuracy subject to fairness constraints
    apply_accuracy_constraint = 0
    sep_constraint = 0
    compute_term = True

    '''find the sensitive feature to specify \phi in balanced surrogate'''
    SENSITIVE_INDEX = df.columns.values.tolist().index('age')
    A_C_INDEX = np.array(x_train[:,SENSITIVE_INDEX]==0).astype(int) # equals 1 if in A+C. equals 0, in B+D
    B_D_INDEX = np.array(x_train[:,SENSITIVE_INDEX]==1).astype(int)
    N = x_train.shape[0]
    A_C_INDEX_TEST = np.array(x_test[:,SENSITIVE_INDEX]==0).astype(int)
    B_D_INDEX_TEST = np.array(x_test[:,SENSITIVE_INDEX]==1).astype(int)
    N_TEST = x_test.shape[0]

    for i in range(0,1):
        temp = i/1000
        sensitive_attrs_to_cov_thresh = {'age': temp}
        # print(i)
        print("\n")
        print("== Balanced  Surrogate ==")
        w_f_cons, p_f_cons, acc_f_cons  = balanced_sensitive_train_test_classifier()

        # to prevent some special classifier
        # for example, assigning positive prediction to all instances with a specific label
        # if acc_f_cons > 0.67:
        vio_list_balance.append(p_f_cons)
        acc_list_balance.append(acc_f_cons)
        method_list_balance.append('B-General Sigmoid')

        
    # acc_list = [float('{:.7f}'.format(i)) for i in acc_list]
    # vio_list = [float('{:.7f}'.format(i)) for i in vio_list]
    # print("\n")
    # print(acc_list)
    # print("\n")
    # print(vio_list)
    # print("\n")
    # print(method_list)




    ''' 3. General Sigmoid'''
    
    apply_fairness_constraints = 1 # set this flag to one since we want to optimize accuracy subject to fairness constraints
    apply_accuracy_constraint = 0
    sep_constraint = 0
    compute_term = True

    # choose the selected parameter w
    w_final = 2
    width = w_final

    for i in range(0,1):
        temp = i/2000
        sensitive_attrs_to_cov_thresh = {'age':temp}
        # print(i)
        print("\n")
        print("== General Sigmoid ==")
        w_f_cons, p_f_cons, acc_f_cons  = general_sigmoid_train_test_classifier()

        # to prevent some special classifier
        # for example, assigning positive prediction to all instances with a specific label
        # if acc_f_cons > 0.65 and p_f_cons < 0.01:
        vio_list_general_sigmoid.append(p_f_cons)
        acc_list_general_sigmoid.append(acc_f_cons)
        method_list_general_sigmoid.append("General Sigmoid")
