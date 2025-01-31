import numpy as np
from tqdm import tqdm
import math

def mi_between_cat_vecs(vec_1, vec_2):
    assert len(vec_1) == len(vec_2)
    n = len(vec_1)
    cats_1 = list(np.unique(vec_1))
    cats_2 = list(np.unique(vec_2))
    cats_1.sort()
    cats_2.sort()
    key_to_ind_1 = dict((key, j) for j, key in enumerate(cats_1))
    key_to_ind_2 = dict((key, j) for j, key in enumerate(cats_2))

    # individual variables
    counts_dict_1 = {}
    for key in cats_1:
        counts_dict_1[key] = np.count_nonzero(vec_1 == key)
    counts_dict_2 = {}
    for key in cats_2:
        counts_dict_2[key] = np.count_nonzero(vec_2 == key)

    counts_vec_1 = np.array([counts_dict_1[key] for key in cats_1])
    counts_vec_2 = np.array([counts_dict_2[key] for key in cats_2])

    freq_vec_1 = counts_vec_1 / n
    freq_vec_2 = counts_vec_2 / n

    # joint variable
    counts_dict = {}
    for j_1 in cats_1:
        for j_2 in cats_2:
            counts_dict[(j_1, j_2)] = 0
    for j_1, j_2 in zip(vec_1, vec_2):
        counts_dict[(j_1, j_2)] += 1

    key_list = list(counts_dict.keys())
    key_to_ind = {}
    for key in key_list:
        key_to_ind[key] = key_list.index(key)
    counts_vec = np.array([counts_dict[key] for key in key_list])
    freq_vec = counts_vec / n

    log_list = []
    for j_1, j_2 in key_list:
        p_1 = freq_vec_1[key_to_ind_1[j_1]]
        p_2 = freq_vec_2[key_to_ind_2[j_2]]
        p_joint = freq_vec[key_to_ind[(j_1, j_2)]]
        if (p_1 == 0.0) or (p_2 == 0.0) or (p_joint == 0.0):
            log_list.append(0.0)
        else:
            log_list.append(np.log(p_joint) - np.log(p_1) - np.log(p_2))
    log_vec = np.array(log_list)
    return np.dot(freq_vec, log_vec)

def pairwise_mi_matrix_from_cat_mats(X_1, X_2):
    n_1, d_1 = X_1.shape
    n_2, d_2 = X_2.shape
    try:
        assert n_1 == n_2
    except AssertionError:
        raise AssertionError(str((n_1, n_2)))
    mi_mat = np.zeros((d_1, d_2))
    for j in tqdm(range(d_1)):
        vec_1 = X_1[:, j]
        for k in range(d_2):
            vec_2 = X_2[:, k]
            mi = mi_between_cat_vecs(vec_1, vec_2)
            mi_mat[j, k] = mi
    return mi_mat

def third_order_ii_between_cat_vecs(vec_1, vec_2, vec_3):
    len_list = [len(vec) for vec in [vec_1, vec_2, vec_3]]
    assert len(np.unique(len_list)) == 1
    # mutual information between first and second cat_vecs
    mi_12 = mi_between_cat_vecs(vec_1, vec_2)
    # compute various entropy terms to compute conditional
    # mutual information between 1 and 2 given 3
    h_13 = second_order_entropy_between_cat_vecs(vec_1,
						vec_3)
    h_23 = second_order_entropy_between_cat_vecs(vec_2,
						vec_3)
    h_123 = third_order_entropy_between_cat_vecs(vec_1,
						vec_2,
						vec_3)
    h_3 = mi_between_cat_vecs(vec_3, vec_3)
    # compute mutual information between 1 and 2 given 3
    mi_12_3 = h_13 + h_23 - h_123 - h_3
    return mi_12 - mi_12_3

def cond_third_order_ii_between_cat_vecs_and_cond_vec(vec_1, vec_2, vec_3, vec_cond):
    len_list = [len(vec) for vec in [vec_1, vec_2, vec_3, vec_cond]]
    assert len(np.unique(len_list)) == 1
    # Construct the conditional interaction information entirely out of joint entropies
    h_14 = second_order_entropy_between_cat_vecs(vec_1, vec_cond)
    h_24 = second_order_entropy_between_cat_vecs(vec_2, vec_cond)
    h_34 = second_order_entropy_between_cat_vecs(vec_3, vec_cond)
    h_124 = third_order_entropy_between_cat_vecs(vec_1, vec_2, vec_cond)
    h_134 = third_order_entropy_between_cat_vecs(vec_1, vec_3, vec_cond)
    h_234 = third_order_entropy_between_cat_vecs(vec_2, vec_3, vec_cond)
    h_1234 = fourth_order_entropy_between_cat_vecs(vec_1, vec_2, vec_3, vec_cond)
    h_4 = mi_between_cat_vecs(vec_cond, vec_cond)
    return h_14 + h_24 + h_34 - h_124 - h_134 - h_234 + h_1234 - h_4

def cond_third_order_ii_between_cat_mats_and_cond_vec(X_1, X_2, X_3, vec_tar):
    n_1, d_1 = X_1.shape
    n_2, d_2 = X_2.shape
    n_3, d_3 = X_3.shape
    n_tar = len(vec_tar)
    n_list = [n_1, n_2, n_3, n_tar]
    assert len(np.unique(n_list)) == 1

    cond_ii_arr = np.zeros((d_1, d_2, d_3), dtype=np.float64)
    for j_1 in tqdm(range(d_1)):
        vec_1 = X_1[:, j_1]
        for j_2 in range(d_2):
            vec_2 = X_2[:, j_2]
            for j_3 in range(d_3):
                vec_3 = X_3[:, j_3]
                ii = cond_third_order_ii_between_cat_vecs_and_cond_vec(vec_1,
                                                    vec_2, vec_3, vec_tar)
                cond_ii_arr[j_1, j_2, j_3] = ii
    return cond_ii_arr

def cond_triplewise_ii_between_cat_mats_and_cond_vec_on_feat(X_1, X_2, X_3, vec_tar):
    n_1, d_1 = X_1.shape
    n_2, d_2 = X_2.shape
    n_3, d_3 = X_3.shape
    n_tar = len(vec_tar)
    n_list = [n_1, n_2, n_3, n_tar]
    assert len(np.unique(n_list)) == 1

    cond_ii_arr = np.zeros((d_1, d_2, d_3), dtype=np.float64)
    for j_1 in tqdm(range(d_1)):
        vec_1 = X_1[:, j_1]
        for j_2 in range(d_2):
            vec_2 = X_2[:, j_2]
            for j_3 in range(d_3):
                vec_3 = X_3[:, j_3]
                ii = cond_third_order_ii_between_cat_vecs_and_cond_vec_on_feat(vec_1,
                                                    vec_2, vec_tar, vec_3)

def triplewise_ii_array_from_cat_mats(X_1, X_2, X_3):
    n_1, d_1 = X_1.shape
    n_2, d_2 = X_2.shape
    n_3, d_3 = X_3.shape
    try:
        n_list = [n_1, n_2, n_3]
        assert len(np.unique(n_list)) == 1
    except AssertionError:
        raise AssertionError(str(n_1, n_2, n_3))
    ii_arr = np.zeros((d_1, d_2, d_3))
    for j in tqdm(range(d_1)):
        vec_1 = X_1[:, j]
        for k in range(d_2):
            vec_2 = X_2[:, k]
            for l in range(d_3):
                vec_3 = X_3[:, l]
                ii = third_order_ii_between_cat_vecs(vec_1,
                                                vec_2,
                                                vec_3)
                ii_arr[j, k, l] = ii
    return ii_arr

def second_order_entropy_between_cat_vecs(vec_1, vec_2):
    assert len(vec_1) == len(vec_2)
    n = len(vec_1)
    cats_1 = list(np.unique(vec_1))
    cats_2 = list(np.unique(vec_2))
    cats_1.sort()
    cats_2.sort()

    # joint variable
    counts_dict = {}
    for j_1 in cats_1:
        for j_2 in cats_2:
            counts_dict[(j_1, j_2)] = 0
    for j_1, j_2 in zip(vec_1, vec_2):
        counts_dict[(j_1, j_2)] += 1

    key_list = list(counts_dict.keys())
    key_to_ind = {}
    for key in key_list:
        key_to_ind[key] = key_list.index(key)
    counts_vec = np.array([counts_dict[key] for key in key_list])
    freq_vec = counts_vec / n

    log_vec = np.array([(np.log(freq) if freq != 0.0 else 0.0) for freq in freq_vec])
    return -np.dot(freq_vec, log_vec)

def third_order_entropy_between_cat_vecs(vec_1, vec_2, vec_3):
    len_list = [len(vec) for vec in [vec_1, vec_2, vec_3]]
    assert len(np.unique(len_list)) == 1
    n = len_list[0]
    cats_1 = list(np.unique(vec_1))
    cats_2 = list(np.unique(vec_2))
    cats_3 = list(np.unique(vec_3))
    cats_1.sort()
    cats_2.sort()
    cats_3.sort()

    # joint variable
    counts_dict = {}
    for j_1 in cats_1:
        for j_2 in cats_2:
            for j_3 in cats_3:
                counts_dict[(j_1, j_2, j_3)] = 0
    for j_1, j_2, j_3 in zip(vec_1, vec_2, vec_3):
        counts_dict[(j_1, j_2, j_3)] += 1

    key_list = list(counts_dict.keys())
    key_to_ind = {}
    for key in key_list:
        key_to_ind[key] = key_list.index(key)
    counts_vec = np.array([counts_dict[key] for key in key_list])
    freq_vec = counts_vec / n

    log_vec = np.array([(np.log(freq) if freq != 0.0 else 0.0) for freq in freq_vec])
    return -np.dot(freq_vec, log_vec)

def fourth_order_entropy_between_cat_vecs(vec_1, vec_2, vec_3, vec_4):
    len_list = [len(vec) for vec in [vec_1, vec_2, vec_3, vec_4]]
    assert len(np.unique(len_list)) == 1
    n = len_list[0]
    cats_1 = list(np.unique(vec_1))
    cats_2 = list(np.unique(vec_2))
    cats_3 = list(np.unique(vec_3))
    cats_4 = list(np.unique(vec_4))
    cats_1.sort()
    cats_2.sort()
    cats_3.sort()
    cats_4.sort()

    # joint variable
    counts_dict = {}
    for j_1 in cats_1:
        for j_2 in cats_2:
            for j_3 in cats_3:
                for j_4 in cats_4:
                    counts_dict[(j_1, j_2, j_3, j_4)] = 0
    for j_1, j_2, j_3, j_4 in zip(vec_1, vec_2, vec_3, vec_4):
        counts_dict[(j_1, j_2, j_3, j_4)] += 1

    key_list = list(counts_dict.keys())
    key_to_ind = {}
    for key in key_list:
        key_to_ind[key] = key_list.index(key)
    counts_vec = np.array([counts_dict[key] for key in key_list])
    freq_vec = counts_vec / n

    log_vec = np.array([(np.log(freq) if freq != 0.0 else 0.0) for freq in freq_vec])
    return -np.dot(freq_vec, log_vec)

def cond_mi_between_cat_vecs_and_cond_vec(vec_1, vec_2, vec_cond):
    len_list = [len(vec) for vec in [vec_1, vec_2, vec_cond]]
    assert len(np.unique(len_list)) == 1
    n = len_list[0]
    cats_1 = list(np.unique(vec_1))
    cats_2 = list(np.unique(vec_2))
    cats_cond = list(np.unique(vec_cond))

    ent_1_cond = second_order_entropy_between_cat_vecs(vec_1, vec_cond)
    ent_2_cond = second_order_entropy_between_cat_vecs(vec_2, vec_cond)
    ent_cond = mi_between_cat_vecs(vec_cond, vec_cond)

    # joint variable
    joint_list = []
    obs_list = []
    key_to_ind = {}
    for pair in zip(vec_1, vec_2):
        try:
            joint_list.append(key_to_ind[pair])
        except KeyError:
            obs_list.append(pair)
            key_to_ind[pair] = obs_list.index(pair)
            joint_list.append(key_to_ind[pair])
    joint_vec = np.array(joint_list)
    ent_joint_cond = second_order_entropy_between_cat_vecs(joint_vec, vec_cond)
    return ent_1_cond + ent_2_cond - ent_cond - ent_joint_cond

def pairwise_cond_mi_matrix_from_cat_mats_and_cond_vec(X_1, X_2, vec_cond):
    n_1, d_1 = X_1.shape
    n_2, d_2 = X_2.shape
    try:
        assert n_1 == n_2
    except AssertionError:
        raise AssertionError(str((n_1, n_2)))
    cond_mi_mat = np.zeros((d_1, d_2))
    for j in tqdm(range(d_1)):
        vec_1 = X_1[:, j]
        for k in range(d_2):
            vec_2 = X_2[:, k]
            cond_mi = cond_mi_between_cat_vecs_and_cond_vec(vec_1, vec_2, vec_cond)
            cond_mi_mat[j, k] = cond_mi
    return cond_mi_mat

def pairwise_cond_mi_matrix_for_qubo(X_1, X_2, vec_tar):
    n_1, d_1 = X_1.shape
    n_2, d_2 = X_2.shape
    assert n_1 == n_2
    cond_mi_mat = np.zeros((d_1, d_2))
    for j in tqdm(range(d_1)):
        vec_1 = X_1[:, j]
        for k in range(d_2):
            vec_2 = X_2[:, k]
            cond_mi = cond_mi_between_cat_vecs_and_cond_vec(vec_1, vec_tar, vec_2)
            cond_mi_mat[j, k] = cond_mi
    return cond_mi_mat

def identity_a(vec_a, vec_b):
    mi = mi_between_cat_vecs(vec_a, vec_b)
    ent_a = mi_between_cat_vecs(vec_a, vec_a)
    ent_b = mi_between_cat_vecs(vec_b, vec_b)
    ent_ab = second_order_entropy_between_cat_vecs(vec_a, vec_b)
    return mi, (ent_a + ent_b - ent_ab)

def identity_b(vec_a, vec_b, vec_c):
    mi_ab_c = cond_mi_between_cat_vecs_and_cond_vec(vec_a, vec_b, vec_c)
    ent_ac = second_order_entropy_between_cat_vecs(vec_a, vec_c)
    ent_bc = second_order_entropy_between_cat_vecs(vec_b, vec_c)
    ent_abc = third_order_entropy_between_cat_vecs(vec_a, vec_b, vec_c)
    ent_c = mi_between_cat_vecs(vec_c, vec_c)
    return mi_ab_c, (ent_ac + ent_bc - ent_abc - ent_c)

def identity_c(vec_a, vec_b, vec_c):
    ii_abc = third_order_ii_between_cat_vecs(vec_a, vec_b, vec_c)
    mi_ab = mi_between_cat_vecs(vec_a, vec_b)
    mi_ab_c = cond_mi_between_cat_vecs_and_cond_vec(vec_a, vec_b, vec_c)
    return ii_abc, (mi_ab - mi_ab_c)

def dist_mat_from_mi_mat(mi_mat):
    dist_mat = np.zeros(mi_mat.shape)
    m, n = mi_mat.shape
    for j in tqdm(range(m)):
        for k in range(j+1, n):
            dist_mat[j, k] = mi_mat[j, j] + mi_mat[k, k] - 2*mi_mat[j, k]
            dist_mat[k, j] = dist_mat[j, k]
    return dist_mat

def dist_mat_from_three_mi_mats(mi_mat_both, mi_mat_self_1, mi_mat_self_2):
    dist_mat = np.zeros(mi_mat_both.shape)
    m, n = mi_mat_both.shape
    for j in tqdm(range(m)):
        for k in range(n):
            dist_mat[j, k] = mi_mat_self_1[j, j] + mi_mat_self_2[k, k] - 2*mi_mat_both[j, k]
    return dist_mat

def normalized_dist_mat_from_mi_and_dist_mats(mi_mat, dist_mat):
    norm_mat = np.zeros(mi_mat.shape)
    m, n = mi_mat.shape
    for j in tqdm(range(m)):
        ent_j = mi_mat[j, j]
        for k in range((j+1), n):
            ent_k = mi_mat[k, k]
            dist = dist_mat[j, k]
            joint_ent = (dist + ent_j + ent_k) / 2
            norm_mat[j, k] = dist / joint_ent
            norm_mat[k, j] = dist / joint_ent
    return norm_mat

# ---------------------han stats section-----------------------------

#def first_order_entropy(vec):
#	# check input?
#	entropy = 0
#	for i in vec:
#		entropy -= i * math.log(i)
#		
#	return entropy


def second_order_han(relevant_vecs):
	# k: number of variables included in entropy
	# entropy: entropy of RV's 2,1 through 2,k
	
	#k = len(relevant_vecs)
	n = len(relevant_vecs)
	# ensure math.comb(n, 2) nonzero.
	#dc_const = k / (math.comb(n, k))
	dc_const = 1 / (2 * math.comb(n, 2))
	
	# do sum: range starts at 0, ends at k-1.
	entropy_sum = 0
#	for i in range(k):
#		for j in range(k):
#			if i==j:
#				entropy_sum += 0
#			else:
	for i in range(n):
		for j in range(i+1, n):
			entropy_sum += second_order_entropy_between_cat_vecs(relevant_vecs[i], relevant_vecs[j])

	return entropy_sum * dc_const
	

def second_order_han_cond(relevant_vecs):
	# k: number of variables included in entropy
	# entropy: entropy of RV's 2,1 through 2,k
	# the last "relevant" vector is the conditional one (Y) - should be 3 total

	k = len(relevant_vecs) - 1

	# ensure math.comb(n, 2) nonzero.
#	dc_const = 1 / (k*(math.comb(k, 2)))
	dc_const = 1 / (2*(math.comb(k, 2)))
	
	# do sum: range starts at 0, ends at k-1.
	entropy_sum = 0
#	for i in range(k):
#		for j in range(k):
#			if i==j:
#				entropy_sum += 0
#			else:
	for i in range(k):
		for j in range(i+1, k):
			entropy_sum += third_order_entropy_between_cat_vecs(relevant_vecs[i], relevant_vecs[j], relevant_vecs[k])
#			entropy_sum -= first_order_entropy(relevant_vecs[k])
			entropy_sum -= mi_between_cat_vecs(relevant_vecs[k],
						relevant_vecs[k])

	return entropy_sum * dc_const


def third_order_han(relevant_vecs):
	# k: number of variables included in entropy
	# entropy: entropy of RV's 2,1 through 2,k

	k = len(relevant_vecs)

	# ensure math.comb(n, 2) nonzero.
#	dc_const = 1 / (k*(math.comb(k, 3)))
	dc_const = 1 / (3 * (math.comb(k, 3)))
	
	# do sum: range starts at 0, ends at k-1.
	entropy_sum = 0
#	for i in range(k):
#		for j in range(k):
#			for l in range(k):
#				if i==j or j==k or i==l:
#					entropy_sum += 0
#				else:
	for i in range(k):
		for j in range(i+1, k):
			for l in range(j+1, k):
				entropy_sum += third_order_entropy_between_cat_vecs(relevant_vecs[i], relevant_vecs[j], relevant_vecs[l])

	return entropy_sum * dc_const
	
	
def third_order_han_cond(relevant_vecs):
	# k: number of variables included in entropy
	# entropy: entropy of RV's 2,1 through 2,k
	# relevant_vecs has 4 vectors in it - the last one is the conditioning vector

	k = len(relevant_vecs) - 1

	# ensure math.comb(n, 2) nonzero.
#	dc_const = 1 / (k*(math.comb(k, 2)))
	dc_const = 1 / (3*(math.comb(k, 3)))
	
	# do sum: range starts at 0, ends at k-1.
	entropy_sum = 0
#	for i in range(k):
#		for j in range(k):
#			for l in range(k):
#				if i==j or j==k or i==l:
#					entropy_sum += 0
#				else:
	for i in range(k):
		for j in range(i+1, k):
			for l in range(j+1, k):
#				entropy_sum += fourth_order_entropy_between_cat_vecs(relevant_vecs[i], relevant_vecs[j], relevant_vecs[l], relevant_vecs[k])
				entropy_sum += fourth_order_entropy_between_cat_vecs(relevant_vecs[i], relevant_vecs[j], relevant_vecs[l], relevant_vecs[k])
				#entropy_sum -= first_order_entropy(relevant_vecs[k])
				entropy_sum -= mi_between_cat_vecs(relevant_vecs[k],
							relevant_vecs[k])

	return entropy_sum * dc_const


def get_delta_three(relevant_vecs, cond_vec):
#	return third_order_han(relevant_vecs) - third_order_han_cond(relevant_vecs.append(cond_vec))
	return third_order_han(relevant_vecs) - third_order_han_cond(relevant_vecs + [cond_vec])


def get_delta_two(relevant_vecs, cond_vec):
#	return second_order_han(relevant_vecs) - second_order_han(relevant_vecs.append(cond_vec))
	return second_order_han(relevant_vecs) - second_order_han(relevant_vecs + [cond_vec])

