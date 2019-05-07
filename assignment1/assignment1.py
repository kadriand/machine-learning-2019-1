import numpy as np

print('Loading Assignment 1 v1')

#6x5
TD = [[2, 3, 0, 3, 7], [0, 5, 5, 0, 3], [5, 0, 7, 3, 3], [3, 1, 0, 9, 9], [0, 0, 7, 1, 3], [6, 9, 4, 6, 0]]
#1x6
L = [5, 2, 3, 6, 4, 3]

# print(TD)
# print(L)

TDmat=np.mat(TD)
Lmat = np.transpose(np.mat(L))

print(TDmat)
print(Lmat)

def point_a():
	sums=TDmat*np.ones((5,1))
	tot_sum=(np.ones(6)*sums) 
	# print(np.ones(6)*sums) 
	inverted=1/tot_sum[0,0]
	# print(inverted)
	P_d_t=inverted*TDmat
	print("P_d_t")
	print(P_d_t)

def point_b():
	sums=TDmat*np.ones((5,1))
	# print(sums) 
	inverted=1/sums
	# print(inverted)
	# print(inverted.A1)
	norm_factors=np.diag(inverted.A1)
	# print(norm_factors)
	P_D_t=norm_factors*TDmat
	print("P_D_t")
	print(P_D_t)

def point_c():
	sums=np.ones(6)*TDmat
	# print(sums)
	inverted=1/sums
	# print(inverted)
	norm_factors=np.diag(inverted.A1)
	# print(norm_factors)
	P_T_d=TDmat*norm_factors
	print("P_T_d")
	print(P_T_d)

def point_d():
	sums_doc=np.ones(6)*TDmat
	# print(sums_doc)
	tot_sum=sums_doc*np.ones((5,1)) 
	# print(tot_sum)
	norm_factors=1/tot_sum[0,0]
	# print(norm_factors)
	P_D=sums_doc*norm_factors
	print("P_D")
	print(P_D)

def point_e():
	sums_term = TDmat*np.ones((5,1)) 
	# print(sums_term)
	tot_sum=np.ones(6)*sums_term
	# print(tot_sum)
	norm_factors=1/tot_sum[0,0]
	# print(norm_factors)
	P_T=sums_term*norm_factors
	print("P_T")
	print(P_T)


def point_f():
	l_diag= np.diag(Lmat.A1)
	TD_with_L=l_diag*TDmat
	# print(TD_with_L)
	sum_l=TD_with_L*np.ones((5,1))
	tot_sum_l=np.ones(6)*sum_l
	# print(tot_sum_l) 

	sum_t=TDmat*np.ones((5,1))
	tot_sum_t=np.ones(6)*sum_t
	# print(tot_sum_t) 

	E_l=tot_sum_l/tot_sum_t
	print("E_l")
	print(E_l)


def point_g():
	l_diag= np.diag(Lmat.A1)
	TD_with_L=l_diag*TDmat
	# print(TD_with_L)
	sum_l=TD_with_L*np.ones((5,1))
	tot_sum_l=np.ones(6)*sum_l
	# print(tot_sum_l) 

	sum_t=TDmat*np.ones((5,1))
	tot_sum_t=np.ones(6)*sum_t
	# print(tot_sum_t) 

	E_l=tot_sum_l/tot_sum_t
	# print(E_l)

	L_diff=Lmat-E_l[0,0]*np.ones((6,1))
	# print(L_diff)
	L_diff_sq=np.diag(L_diff.A1)*L_diff
	# print(L_diff_sq)
	# print(sum_t)
	var_L_tot=(np.transpose(L_diff_sq)*sum_t)[0,0]/tot_sum_t
	print("var_L_tot")
	print(var_L_tot)


# point_a()
# point_b()
# point_c()
# point_d()
# point_e()
# point_f()
# point_g()