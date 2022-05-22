import numpy as np
import scipy
from scipy import optimize
import pickle

Rounds = 30 # number of independent problem realizations
d = 2  # dimension
T = 10000 # time horizon
C = 1.1 #alpha_safety
alpha_safety = 1.1
theta_star = np.array([-0.1, -1.0])
zeta_star = np.array([0, 1])
alpha = np.array([[0], [0], [4], [alpha_safety]])
Zeta_else = np.array([[0, -1], [-1, 1], [1, 1]])
result = scipy.optimize.linprog(theta_star, np.vstack((Zeta_else, zeta_star)), alpha, None, None, (None, None), method= 'simplex')

x_star = result.x
value = result.fun
optimal_slack = result.slack

S = np.linalg.norm(theta_star)
R = 0.1
L = 4
Lambda = 1

delta = 1/4/T
epsilon = 0.1  # slackness
# all of the above are global variables

# output we need for the plots: for LTS and DO, the regret w.r.t reward and safety and basic indices picked
obj_regret_Linfty = np.zeros([Rounds, T])
obj_regret_L1 = np.zeros([Rounds, T])
cst_regret_Linfty = np.zeros([Rounds, T])
cst_regret_L1 = np.zeros([Rounds, T])
BIS0 = np.zeros([Rounds, T]) # bis1 is 1 if the corresponding constraint is active
BIS1 = np.zeros([Rounds, T])
BIS0_i = np.zeros([Rounds, T]) # for L1 confidence sets
BIS1_i = np.zeros([Rounds, T])


for r in range(Rounds):
    noise = np.random.normal(0, R, (d, T))

    V = Lambda * np.eye(d)
    inv_V = 1/Lambda * np.eye(d)
    V_i = Lambda * np.eye(d) #i for LTS related values
    inv_V_i = 1/Lambda * np.eye(d)

    X = np.zeros([T, d]) # matrix for x_t played
    Y1 = np.zeros([T, 1]) # matrix for theta x_t observed
    Y2 = np.zeros([T, 1]) # matrix for zeta x_t observed

    # for LTS
    X_i = np.zeros([T, d]) # matrix for x_t played
    Y1_i = np.zeros([T, 1]) # matrix for theta x_t observed
    Y2_i = np.zeros([T, 1]) # matrix for zeta x_t observed

    theta_hat = np.zeros([1, d])
    theta_hat_mat = np.zeros([T, d])
    theta_tilde_mat = np.zeros([T, d])
    zeta_hat = np.zeros([1, d])
    zeta_hat_mat = np.zeros([T, d])
    zeta_tilde_mat = np.zeros([T, d])
    regret = np.zeros([T, 1])
    slacked_regret = np.zeros([T, 1])
    slack_mat = np.zeros([T, alpha.shape[0]]) # slack for each constraint

    #for LTS
    theta_hat_i = np.zeros([1, d])
    theta_hat_mat_i = np.zeros([T, d])
    theta_tilde_mat_i = np.zeros([T, d])
    zeta_hat_i = np.zeros([1, d])
    zeta_hat_mat_i = np.zeros([T, d])
    zeta_tilde_mat_i = np.zeros([T, d])
    Sample = np.random.normal(0, 1+ 2 * L * S / C, (d, T)) # to store the TS sample
    regret_i = np.zeros([T, 1])
    slacked_regret_i = np.zeros([T, 1])
    slack_mat_i = np.zeros([T, alpha.shape[0]]) # slack for each constraint


    for t in range(T):
        # for Linfty
        beta = R * np.sqrt(2 * np.log(2 * np.sqrt(np.linalg.det(V)) / delta / np.sqrt(Lambda ** d))) + np.sqrt(Lambda) * S
        X_candidate = np.zeros([(2 * d) ** 2, d])
        val_candidate = np.zeros((2 * d) ** 2)
        theta_candidate = np.zeros([(2 * d) ** 2, d])
        zeta_candidate = np.zeros([(2 * d) ** 2, d])
        slack_candidate = np.zeros([(2 * d) ** 2, alpha.shape[0]])
        a = np.dot(scipy.linalg.sqrtm(inv_V), np.array([[-beta, -beta, beta, beta], [-beta, beta, -beta, -beta]])) # L infty confidence set
        for i in range(2 * d):
            for j in range(2 * d):
                theta = a[:, i] + theta_hat.reshape(1,- 1)
                zeta = a[:, j] + zeta_hat.reshape(1, -1)
                solution = scipy.optimize.linprog(theta, np.vstack((Zeta_else, zeta)), alpha, None, None, (None, None), method='simplex')
                if solution.success:
                    X_candidate[j * d * 2 + i, :] = solution.x
                    val_candidate[j * d * 2 + i] = solution.fun
                    theta_candidate[j * d * 2 + i, :] = theta
                    zeta_candidate[j * d * 2 + i, :] = zeta
                    slack_candidate[j * d * 2 + i, :] = solution.slack
        M = np.amin(val_candidate)
        I = np.argmin(val_candidate)
        theta_tilde = theta_candidate[I, :]
        zeta_tilde = zeta_candidate[I, :]
        x = X_candidate[I, :]
        slack = slack_candidate[I, :]
        # observations
        y1 = np.dot(x, theta_star) + noise[0, t]
        y2 = np.dot(x, zeta_star) + noise[1, t]
        # updates
        X[t, :] = x
        Y1[t, 0] = y1
        Y2[t, 0] = y2
        inv_V = inv_V - 1 / (1 + np.dot(x, np.dot(inv_V, x))) * np.outer(np.dot(inv_V, x), np.dot(inv_V, x))
        V = V + np.outer(x, x)
        theta_hat = np.transpose(np.matmul(inv_V, np.matmul(np.transpose(X[:t + 1, :]), Y1[: t+1, :])))
        theta_hat_mat[t, :] = theta_hat
        zeta_hat = np.transpose(np.matmul(inv_V, np.matmul(np.transpose(X[:t + 1, :]), Y2[: t+1, :])))
        zeta_hat_mat[t, :] = zeta_hat
        theta_tilde_mat[t, :] = theta_tilde
        zeta_tilde_mat[t, :] = zeta_tilde
        slack_mat[t, :] = slack
        regret[t] = np.max([np.dot(theta_star, x - x_star), np.dot(zeta_star, x) - alpha_safety, 0])
        slacked_regret[t] = np.max([np.dot(theta_star, x - x_star), np.dot(zeta_star, x) - alpha_safety - epsilon, 0])

        # for L1
        beta_i = (R * np.sqrt(2 * np.log(2 * np.sqrt(np.linalg.det(V_i)) / delta / np.sqrt(Lambda ** d))) + np.sqrt(Lambda) * S) * np.sqrt(d)
        X_candidate_i = np.zeros([(2 * d) ** 2, d])
        val_candidate_i = np.zeros((2 * d) ** 2)
        theta_candidate_i = np.zeros([(2 * d) ** 2, d])
        zeta_candidate_i = np.zeros([(2 * d) ** 2, d])
        slack_candidate_i = np.zeros([(2 * d) ** 2, alpha.shape[0]])
        # l_1 confidence set
        a_i = np.dot(scipy.linalg.sqrtm(inv_V_i),np.array([[0, 0, -beta_i, beta_i], [-beta_i, beta_i, 0, 0]]))
        for i in range(2 * d):
            for j in range(2 * d):
                theta_i = a_i[:, i] + theta_hat_i.reshape(1, - 1)
                zeta_i = a_i[:, j] + zeta_hat_i.reshape(1, -1)
                solution_i = scipy.optimize.linprog(theta_i, np.vstack((Zeta_else, zeta_i)), alpha, method='simplex')
                if solution_i.success:
                    X_candidate_i[j * d * 2 + i, :] = solution_i.x
                    val_candidate_i[j * d * 2 + i] = solution_i.fun
                    theta_candidate_i[j * d * 2 + i, :] = theta_i
                    zeta_candidate_i[j * d * 2 + i, :] = zeta_i
                    slack_candidate_i[j * d * 2 + i, :] = solution_i.slack
        M = np.amin(val_candidate_i)
        I = np.argmin(val_candidate_i)
        theta_tilde_i = theta_candidate_i[I, :]
        zeta_tilde_i = zeta_candidate_i[I, :]
        x_i = X_candidate_i[I, :]
        slack_i = slack_candidate_i[I, :]
        # observations
        y1_i = np.dot(x_i, theta_star) + noise[0, t]
        y2_i = np.dot(x_i, zeta_star) + noise[1, t]
        # updates
        X_i[t, :] = x_i
        Y1_i[t, 0] = y1_i
        Y2_i[t, 0] = y2_i
        inv_V_i = inv_V_i - 1 / (1 + np.dot(x_i, np.dot(inv_V_i, x_i))) * np.outer(np.dot(inv_V_i, x_i), np.dot(inv_V_i, x_i))
        V_i = V_i + np.outer(x_i, x_i)
        theta_hat_i = np.transpose(np.matmul(inv_V_i, np.matmul(np.transpose(X_i[:t + 1, :]), Y1_i[: t + 1, :])))
        theta_hat_mat_i[t, :] = theta_hat_i
        zeta_hat_i = np.transpose(np.matmul(inv_V_i, np.matmul(np.transpose(X_i[:t + 1, :]), Y2_i[: t + 1, :])))
        zeta_hat_mat_i[t, :] = zeta_hat_i
        theta_tilde_mat_i[t, :] = theta_tilde_i
        zeta_tilde_mat_i[t, :] = zeta_tilde_i
        slack_mat_i[t, :] = slack_i
        regret_i[t] = np.max([np.dot(theta_star, x_i - x_star), np.dot(zeta_star, x_i) - alpha_safety, 0])
        slacked_regret_i[t] = np.max([np.dot(theta_star, x_i - x_star), np.dot(zeta_star, x_i) - alpha_safety - epsilon, 0])
        if t % 200 == 0:
            print('r =', r, 't =', t, 'x =', x, 'x_i =', x_i)

    # data storage
    obj_regret_Linfty[r, :] = np.dot(X, theta_star) - np.dot(x_star, theta_star)
    obj_regret_L1[r, :] = np.dot(X_i, theta_star) - np.dot(x_star, theta_star)
    cst_regret_Linfty[r, :] = np.dot(X, zeta_star) - alpha_safety
    cst_regret_L1[r, :] = np.dot(X_i, zeta_star) - alpha_safety
    indices = slack_mat[:, np.where(optimal_slack == 0)[0]]
    BIS0[r, :] = indices[:, 0] == 0
    BIS1[r, :] = indices[:, 1] == 0
    indices_i = slack_mat_i[:, np.where(optimal_slack == 0)[0]]
    BIS0_i[r, :] = indices_i[:, 0] == 0
    BIS1_i[r, :] = indices_i[:, 1] == 0

with open(f'LinftyvsL1.pickle', 'wb') as file:
    pickle.dump((obj_regret_Linfty, obj_regret_L1, cst_regret_Linfty, cst_regret_L1, BIS0, BIS1, BIS0_i, BIS1_i), file)
