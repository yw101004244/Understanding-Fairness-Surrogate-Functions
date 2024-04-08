import os,sys
sys.path.append(r'c:\Users\yao\Desktop\fair-classification-master\fair_classification') # the code for fair classification is in this directory

import numpy as np
from prepare_adult_data import *
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints



def test_adult_data():

	""" Load the adult data """
	X, y, x_control = load_adult_data(load_data_size=None) # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
	ut.compute_violation_of_disparate_impact(x_control["sex"], y) # compute the p-rule in the original data

	
	""" Split the data into train and test """
	X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
	train_fold_size = 0.7
	x_train, y_train, x_control_train, x_test, y_test, x_control_test, split_point = ut.split_into_train_test(X, y, x_control, train_fold_size)

	""" Some definitions of variables"""
	# These 3 lines are preparations for computing the term afterwards
	# After debugging, we know that male is encoded into 0, while 1 denotes female(protected).
	A_C_INDEX = np.array(x_control["sex"][split_point:]==1).astype(int)
	B_D_INDEX = np.array(x_control["sex"][split_point:]==0).astype(int)
	N = len(x_control["sex"][split_point:])
	

	# Some other parameters
	compute_term = False # for unconstrained case, do not need to compute
	apply_fairness_constraints = None
	apply_accuracy_constraint = None
	sep_constraint = None

	loss_function = lf._logistic_loss
	sensitive_attrs = ["sex"]
	sensitive_attrs_to_cov_thresh = {}
	gamma = None

	# the main routine of training
	def train_test_classifier():
		w = ut.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
		train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
		distances_boundary_test = (np.dot(x_test, w)).tolist()
		all_class_labels_assigned_test = np.sign(distances_boundary_test)
		correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
		cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
		violation_of_disparate_impact = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])	
		
		if compute_term == True:
			d_theta = np.dot(w, x_test.T)
			term = ut.compute_term_removed(d_theta, A_C_INDEX, B_D_INDEX, N, violation_of_disparate_impact)
			print("Bound: ", term)

		# draw boxplot
		# ut.draw_boxplot(w, x_test)
		
		return w, violation_of_disparate_impact, test_score









	""" 1. Classify the data while optimizing for accuracy """
	print
	print("== Unconstrained (original) classifier ==")
	# all constraint flags are set to 0 since we want to train an unconstrained (original) classifier
	apply_fairness_constraints = 0
	apply_accuracy_constraint = 0
	sep_constraint = 0
	compute_term = True
	w_uncons, p_uncons, acc_uncons = train_test_classifier()
	






	""" 2. Now classify such that we optimize for accuracy while achieving perfect fairness """
	apply_fairness_constraints = 1 # set this flag to one since we want to optimize accuracy subject to fairness constraints
	apply_accuracy_constraint = 0
	sep_constraint = 0
	compute_term = True
	vio_list = []
	acc_list = []
	method_list = []

	for i in range(1,50,4):
		thres = i / 1000
		print("\n")
		print(thres)
		sensitive_attrs_to_cov_thresh = {"sex":thres}
		print
		print("== Classifier with fairness constraint ==")
		w_f_cons, p_f_cons, acc_f_cons  = train_test_classifier()
		vio_list.append(p_f_cons)
		acc_list.append(acc_f_cons)




	for i in range(len(vio_list)):
		method_list.append('general sigmoid')

	
	acc_list = [float('{:.4f}'.format(i)) for i in acc_list]
	vio_list = [float('{:.4f}'.format(i)) for i in vio_list]
	print(acc_list)
	print(vio_list)
	print(method_list)


	# ut.draw_box_plot_together(w_uncons,w_f_cons,x_test)



	
	# """ 3. Classify such that we optimize for fairness subject to a certain loss in accuracy """
	# apply_fairness_constraints = 0 # flag for fairness constraint is set back to0 since we want to apply the accuracy constraint now
	# apply_accuracy_constraint = 1 # now, we want to optimize fairness subject to accuracy constraints
	# sep_constraint = 0
	# compute_term = True

	# for i in range(2,20,2):
	# 	gamma = i/20 # gamma controls how much loss in accuracy we are willing to incur to achieve fairness -- increase gamme to allow more loss in accuracy
	# 	print(gamma)
	# 	print("== Classifier with accuracy constraint ==")
	# 	w_a_cons, p_a_cons, acc_a_cons = train_test_classifier()	





	""" 
	Classify such that we optimize for fairness subject to a certain loss in accuracy 
	In addition, make sure that no points classified as positive by the unconstrained (original) classifier are misclassified.

	"""
	# apply_fairness_constraints = 0 # flag for fairness constraint is set back to0 since we want to apply the accuracy constraint now
	# apply_accuracy_constraint = 1 # now, we want to optimize accuracy subject to fairness constraints
	# sep_constraint = 1 # set the separate constraint flag to one, since in addition to accuracy constrains, we also want no misclassifications for certain points (details in demo README.md)
	# gamma = 1000.0
	# print("== Classifier with accuracy constraint (no +ve misclassification) ==")
	# w_a_cons_fine, p_a_cons_fine, acc_a_cons_fine  = train_test_classifier()

	return

def main():
	test_adult_data()


if __name__ == '__main__':
	main()