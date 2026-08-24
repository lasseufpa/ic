'''
Examples of entropy, cross-entropy and Kullback–Leibler (KL) divergence.
References:
[1] KL divergence: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
[2] Cross-entropy: https://www.youtube.com/watch?v=ErfnhcEV1O8&ab_channel=Aur%C3%A9lienG%C3%A9ron
[3] Logits nomenclature: https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow
[4] Binary cross-entropy: https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c
'''
import numpy as np
from scipy.special import rel_entr
from matplotlib import pyplot as plt

def binary_cross_entropy(true_probabilities, estimated_probabilities, use_log2 = True):
    '''
    Average binary cross entropy among several binomial PMFs, also called binary cross entropy (BCE).
    '''
    N = len(true_probabilities)
    bces = np.zeros( (N,), dtype=float) #initialize
    for i in range(N): #calculate BCE for each binomial distribution
        bces[i] = individual_binary_cross_entropy(true_probabilities[i], estimated_probabilities[i], use_log2 = use_log2)
    return np.mean(bces)

def individual_binary_cross_entropy(p, q, use_log2 = True):
    '''
    Binary cross entropy (BCE) between a single pair of two binomial distributions.
    p informs about the true Binomial probability distribution, and q about the
    estimated Binomial probability distribution.
    '''
    #create the binomial distribution [p, 1-p] from the probabilities p (the same for q)
    return cross_entropy(np.array([p, 1.0-p]), 
                        np.array([q, 1.0-q]),
                        use_log2 = use_log2)

def individual_binary_cross_entropy_v2(p, q, use_log2 = True, eps=1e-7):
    '''
    Calculates the cross-entropy of a binomial probability mass function (PMF).
    The threshold eps avoids numerical errors caused by probabilities too close
    to zero or one.
    '''
    if p < eps:
        p = eps
    elif p > 1-eps:
        p = 1-eps
    if q < eps:
        q = eps
    elif q > 1-eps:
        q = 1-eps
    if use_log2: #use log_2 and, therefore, entropy in bits as in scikit-learn
        cross_entropy = -p*np.log2(q) - (1-p)*np.log2(1-q)
    else: #use log_e and, therefore, entropy in neper as in Geron's book and tensorflow
        cross_entropy = -p*np.log(q) - (1-p)*np.log(1-q)
    return cross_entropy

def individual_binary_entropy(p, use_log2 = True):
    '''
    Calculates the entropy of a single binomial probability mass function (PMF).
    '''
    if p==0 or p==1:
        return 0    
    if use_log2: #use log_2 and, therefore, entropy in bits as in scikit-learn
        entropy = -p*np.log2(p) - (1-p)*np.log2(1-p)
    else: #use log_e and, therefore, entropy in neper as in Geron's book
        entropy = -p*np.log(p) - (1-p)*np.log(1-p)
    return entropy

def entropy(probability_mass_function, use_log2 = True):
    '''
    Calculates the entropy of a probability mass function (PMF).
    '''
    zero_prob_indices = np.argwhere(probability_mass_function == 0)
    probability_mass_function[zero_prob_indices] = 10 #use any number, just to avoid numerical errors
    if use_log2: #use log_2 and, therefore, entropy in bits as in scikit-learn
        information = np.log2(1.0 / probability_mass_function)
    else: #use log_e and, therefore, entropy in neper as in Geron's book
        information = np.log(1.0 / probability_mass_function)
    probability_mass_function[zero_prob_indices] = 0 #restore zero values
    information[zero_prob_indices] = 0 #the limit is zero because 0*log(1/0) -> 0 (y=log(x) grows slower than y=x)    
    weighted_information  = probability_mass_function * information    
    entropy = np.sum(weighted_information)
    return entropy 

def cross_entropy(true_pmf, estimated_pmf, use_log2 = True):
    '''
    Calculates cross-entropy between two PMFs.
    '''
    N = len(true_pmf)
    assert(N == len(estimated_pmf))
    cross_entropy = np.zeros( (N,), dtype=float)
    for i in range(N):
        if (true_pmf[i] > 0) and (estimated_pmf[i] > 0):
            if use_log2: #use log_2 and, therefore, entropy in bits as in scikit-learn
                cross_entropy[i] = true_pmf[i]*np.log2(1.0 / estimated_pmf[i])
            else: #use log_e and, therefore, entropy in neper as in Geron's book
                cross_entropy[i] = true_pmf[i]*np.log(1.0 / estimated_pmf[i])
        elif (true_pmf[i] == 0) and (estimated_pmf[i] == 0):
            cross_entropy[i] = 0
        elif (true_pmf[i] == 0) and (estimated_pmf[i] > 0):
            cross_entropy[i] = 0
        else: #true_pmf[i] > 0 and estimated_pmf[i] = 0
            print('Warning in cross-entropy: true_pmf[i] =',true_pmf[i],'and estimated_pmf[i] = ',estimated_pmf[i])
            cross_entropy[i] = np.Inf
    return np.sum(cross_entropy)

def one_hot_cross_entropy(true_pmf, estimated_pmf, use_log2 = True):
    '''
    Calculates cross-entropy assuming true_pmf is using one-hot encoding (all elements
    are 0, but one of them that is 1).
    Note that the values of elements in estimated_pmf that do not correspond to
    the position indicated by the correct label.
    '''
    #check if one-hot encoding is being used
    zero_prob_indices = np.argwhere(true_pmf == 0)
    if len(zero_prob_indices)+1 != len(true_pmf):
        raise Exception("PMF in first argument is not using one-hot encoding!")
    class_index = np.argwhere(true_pmf == 1)
    if true_pmf[class_index] == 0 and estimated_pmf[class_index] == 0:
        #the limit is zero because 0*log(1/0) -> 0 (y=log(x) grows slower than y=x)    
        return 0.0
    elif true_pmf[class_index] > 0 and estimated_pmf[class_index] == 0:
        return np.Inf
    else:
        ce = -1 #initialize cross-entropy value
        if use_log2: #use log_2 and, therefore, entropy in bits as in scikit-learn
            ce = true_pmf[class_index] * np.log2(1.0 / estimated_pmf[class_index])
        else: #use log_e and, therefore, entropy in neper as in Geron's book
            ce = true_pmf[class_index] * np.log(1.0 / estimated_pmf[class_index])
        return float(ce)

def kl_divergence(true_pmf, estimated_pmf, use_log2 = True):
    '''
    Calculates Kullback–Leibler (KL) divergence between two PMFs.
    '''
    N = len(true_pmf)
    assert(N == len(estimated_pmf))
    kl_divergence_per_element = np.zeros( (N,), dtype=float)
    for i in range(N):
        if (true_pmf[i] > 0) and (estimated_pmf[i] > 0):
            if use_log2: #use log_2 and, therefore, entropy in bits as in scikit-learn
                kl_divergence_per_element[i] = true_pmf[i]*np.log2(true_pmf[i] / estimated_pmf[i])
            else: #use log_e and, therefore, entropy in neper as in Geron's book
                kl_divergence_per_element[i] = true_pmf[i]*np.log(true_pmf[i] / estimated_pmf[i])
        elif (true_pmf[i] == 0) and (estimated_pmf[i] == 0):
            kl_divergence_per_element[i] = 0
        elif (true_pmf[i] == 0) and (estimated_pmf[i] > 0):
            kl_divergence_per_element[i] = 0
        else:
            print('Warning in KL divergence: true_pmf[i] =',true_pmf[i],'and estimated_pmf[i] = ',estimated_pmf[i])            
            kl_divergence_per_element[i] = np.Inf
    return np.sum(kl_divergence_per_element)


def example_entropy_calculation():
    #Compare entropy calculation using natural log and log base 2
    probability_mass_function = np.array([49, 5])/54 #to reproduce pg 177 from Geron's book
    #probability_mass_function = np.array([4, 2])/6 
    entropy_example = entropy(probability_mass_function, use_log2 = False)
    print("Entropy =", entropy_example, "nats")
    entropy_example = entropy(probability_mass_function, use_log2 = True)
    print("Entropy =", entropy_example, "bits")

def example_CE_calculation():
    #Example of https://www.youtube.com/watch?v=ErfnhcEV1O8&ab_channel=Aur%C3%A9lienG%C3%A9ron
    #at time 9 minutes.
    true_pmf = np.array([0, 0, 0, 0, 1, 0, 0])
    estimated_pmf = np.array([0.02, 0.3, 0.45, 0, 0.25, 0.05, 0])
    cross_entropy_example = cross_entropy(true_pmf, estimated_pmf, use_log2 = False)
    print('cross_entropy=',cross_entropy_example,'nats (because we use natural log)')

    #assuming the true PMF is one-hot encoded
    one_hot_CE = one_hot_cross_entropy(true_pmf, estimated_pmf, use_log2 = False)
    print('one_hot_cross_entropy = ', one_hot_CE)

def example_KL_divergence_calculation():
    #Example of https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    true_pmf = np.array([9, 12, 4])/25.0
    estimated_pmf = np.array([1,1,1])/3.0
    kl_divergence_example = kl_divergence(true_pmf, estimated_pmf, use_log2 = False)
    print('kl_divergence=',kl_divergence_example,'nats (because we use natural log)')
    #switch the true and estimated PMFs as done in Wikipedia example:
    kl_divergence_example2 = kl_divergence(estimated_pmf, true_pmf, use_log2 = False)
    print('Switching PMFs, kl_divergence=',kl_divergence_example2,'nats (because we use natural log)')

    #compare two ways of calculating KL divergence
    cross_entropy_example = cross_entropy(true_pmf, estimated_pmf, use_log2 = False)
    entropy_example = entropy(true_pmf, use_log2 = False)
    kl_divergence_example = cross_entropy_example - entropy_example
    print('Using kl_divergence=crossentropy-entropy',kl_divergence_example,'nats (because we use natural log)')

def example_numerical_problems_in_CE():
    # Trigger numerical problem when true_pmf[i]>0 and estimated_pmf[i]=0
    #It corresponds to the situation where your model believes that some class has 
    # zero probability of occurrence, and yet the class pops up in reality. As a result, 
    # the "surprise" of your model is infinitely great: your model did not account for that 
    # event and now needs infinitely many bits to encode it. That is why you get infinity
    # as your cross-entropy.
    print("### Trigger numerical problem:")
    true_pmf = np.array([1, 2, 1])/3.0
    estimated_pmf = np.array([0,1,1])/3.0
    cross_entropy_example = cross_entropy(true_pmf, estimated_pmf, use_log2 = False)
    entropy_example = entropy(true_pmf, use_log2 = False)
    kl_divergence_example = kl_divergence(true_pmf, estimated_pmf, use_log2 = False)
    print('cross_entropy=',cross_entropy_example)
    print('entropy=',entropy_example)
    print('kl_divergence=',kl_divergence_example)
    #using scipy, calculate (P || Q)
    print('kl_divergence in scipy=', sum(rel_entr(true_pmf, estimated_pmf)))

def example_fix_numerical_problems_in_CE():
    # Fix numerical errors of previous example by clipping probabilities
    true_pmf = np.array([1, 2, 1])/3.0
    estimated_pmf = np.array([0,1,1])/3.0
    print("### Fix numerical errors by clipping probabilities")
    EPSILON = 1e-07 # First clip probabilities to stable range
    P_MAX = 1- EPSILON  # 0.9999999
    stable_estimated_pmf = np.clip(estimated_pmf, a_min=EPSILON, a_max=P_MAX)
    cross_entropy_example = cross_entropy(true_pmf, stable_estimated_pmf, use_log2 = False)
    kl_divergence_example = kl_divergence(true_pmf, stable_estimated_pmf, use_log2 = False)
    print('cross_entropy=',cross_entropy_example)
    print('kl_divergence=',kl_divergence_example)
    #using scipy, calculate (P || Q)
    print('kl_divergence in scipy=', sum(rel_entr(true_pmf, stable_estimated_pmf)))

def ce_from_logits():
    #cross entropy from "logits"
    #See reference [3]. TF/Keras call them "logits", but they are simply inputs to softmax
    true_pmf = np.array([0, 1, 0, 0])
    logits = np.array([-18.6, 0.51, 2.94, -12.8]) #strictly, these are not logits
    unnormalized_probabilities = np.exp(logits) 
    estimated_pmf = unnormalized_probabilities/np.sum(unnormalized_probabilities) #softmax
    print("Estimated PMF", estimated_pmf, np.sum(estimated_pmf))
    cross_entropy_example = cross_entropy(true_pmf, estimated_pmf, use_log2 = False)
    print('Our cross entropy implementation=',cross_entropy_example)

def binary_CE():
    #binary cross-entropy (BCE)
    true_pmf = np.array([0, 1, 1, 0])
    logits = np.array([-18.6, 0.51, 2.94, -12.8])
    probabilities_via_sigmoid = 1.0 / (1 + np.exp(-logits) )
    estimated_pmf = probabilities_via_sigmoid
    print("estimated probability for binary PMFs (note that collectively, they do not sum up to 1", estimated_pmf, np.sum(estimated_pmf))

    bce2 = binary_cross_entropy(true_pmf, estimated_pmf, use_log2 = False)
    print("binary cross-entropy (BCE) =", bce2)

def bce_two_ways():
    #BCE again
    p=0.7
    q=0.3
    true_pmf = np.array([p,1-p])
    estimated_pmf = np.array([q,1-q])
    bce2 = binary_cross_entropy(true_pmf, estimated_pmf, use_log2 = False)
    print("binary cross-entropy (BCE) =", bce2)

    bce3 = individual_binary_cross_entropy_v2(p, q, use_log2 = False)
    print("binary cross-entropy (BCE) with another implementation =", bce3)

def plot_graphs():
    #graphs
    M = 1000
    p = np.linspace(0, 1, M)
    entropies = np.zeros( (M,), dtype=float )
    for i in range(len(p)):
        entropies[i]=individual_binary_entropy(p[i], use_log2 = True)
    plt.plot(p, entropies)
    plt.xlabel('Probability p of binomial distribution')
    plt.ylabel('Entropy in bits')

    plt.figure(2)
    p = 1
    q = np.linspace(0, 1, M)
    cross_entropies = np.zeros( (M,), dtype=float )
    mse = np.zeros( (M,), dtype=float )
    for i in range(len(q)):
        cross_entropies[i]=individual_binary_cross_entropy_v2(p, q[i], use_log2 = False)
        mse[i] = (p-q[i])**2
    #plt.semilogy(q, cross_entropies, q, mse)
    plt.plot(q, cross_entropies, q, mse)
    plt.legend(['Cross-entropy', 'MSE'])
    plt.xlabel('Estimated probability q when true prob is p=1')
    plt.ylabel('Loss')

    plt.figure(3)
    p = 0
    q = np.linspace(0, 1, M)
    cross_entropies = np.zeros( (M,), dtype=float )
    mse = np.zeros( (M,), dtype=float )
    for i in range(len(q)):
        cross_entropies[i]=individual_binary_cross_entropy_v2(p, q[i], use_log2 = False)
        mse[i] = (p-q[i])**2
    #plt.semilogy(q, cross_entropies, q, mse)
    plt.plot(q, cross_entropies, q, mse)
    plt.legend(['Cross-entropy', 'MSE'])
    plt.xlabel('Estimated probability q when true prob is p=0')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    print("########### Example 1 ###########")
    example_CE_calculation()
    print("########### Example 2 ###########")
    example_KL_divergence_calculation()
    print("########### Example 3 ###########")
    example_numerical_problems_in_CE()
    print("########### Example 4 ###########")
    example_fix_numerical_problems_in_CE()
    print("########### Example 5 ###########")
    ce_from_logits()
    print("########### Example 6 ###########")
    binary_CE()
    print("########### Example 7 ###########")
    bce_two_ways()
    print("########### Example 8 ###########")
    example_entropy_calculation()
    plot_graphs()