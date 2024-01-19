import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning


class Bandit:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.theta = np.random.randn(n_actions, n_features)

    def get_reward(self, action, x):

        randomness = np.random.normal()
        Reward = (x @ self.theta[action] + randomness)
        Optimal_Reward = np.max(x @ self.theta.T + randomness)
        Worst_Reward = np.min(x @ self.theta.T + randomness)

        # Reward = (Reward - Worst_Reward) / (Optimal_Reward - Worst_Reward)
        # Optimal_Reward = 1

        return Reward , Optimal_Reward, Worst_Reward



class LinUCB:
    def __init__(self, n_actions, n_features, alpha):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha

        # Initialize parameters
        self.A = np.array([np.identity(n_features) for _ in range(n_actions)])  # action covariance matrix

        self.b = np.array([np.zeros(n_features) for _ in range(n_actions)])  # action reward vector

        self.theta = np.array([np.zeros(n_features) for _ in range(n_actions)])  # action parameter vector

    def predict(self, context):
        context = np.array(context)  # Convert list to ndarray
        context = context.reshape(-1, 1)  # reshape the context to a single-column matrix
        p = np.zeros(self.n_actions)
        miu = np.zeros(self.n_actions)
        sigma = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            theta = np.dot(np.linalg.inv(self.A[a]), self.b[a])  # theta_a = A_a^-1 * b_a
            theta = theta.reshape(-1, 1)  # Explicitly reshape theta
            p[a] = np.dot(theta.T, context) + self.alpha * np.sqrt(
                np.dot(context.T, np.dot(np.linalg.inv(self.A[a]), context)))  # p_t(a|x_t) = theta_a^T * x_t + alpha * sqrt(x_t^T * A_a^-1 * x_t)
            miu[a] = np.dot(theta.T, context)
            sigma[a] = np.sqrt(np.dot(context.T, np.dot(np.linalg.inv(self.A[a]), context)))
        return p, miu, sigma

    def update(self, action, context, reward):
        self.A[action] += np.outer(context, context)  # A_a = A_a + x_t * x_t^T
        self.b[action] += reward * context  # b_a = b_a + r_t * x_tx



def Simulator(n_actions, n_features, alpha, Budget, n_steps):
    bandit = Bandit(n_actions, n_features)

    linucb_agent = LinUCB(n_actions, n_features, alpha)

    agents = [linucb_agent]

    cumulative_rewards = np.zeros(n_steps)
    cumulative_regrets = np.zeros(n_steps)

    agent = linucb_agent

    Selected_Actions = np.zeros((n_steps, n_actions))
    Selected_Actions = pd.DataFrame(Selected_Actions)

    Predicted_Rewards = np.zeros((n_steps, n_actions))
    Predicted_Rewards = pd.DataFrame(Predicted_Rewards)

    Predicted_Miu = np.zeros((n_steps, n_actions))
    Predicted_Miu = pd.DataFrame(Predicted_Miu)

    Predicted_Sigma = np.zeros((n_steps, n_actions))
    Predicted_Sigma = pd.DataFrame(Predicted_Sigma)

    Vector = np.zeros((n_steps, 13))
    Vector = pd.DataFrame(Vector)

    for t in range(n_steps):
        # if (t+1) % 500 == 0:
        #     alpha = alpha / 2
        alpha = alpha
        x = np.random.randn(n_features)
        pred_rewards, miu, sigma = agent.predict([x])
        Predicted_Rewards.iloc[t] = pred_rewards
        Predicted_Miu.iloc[t] = miu
        Predicted_Sigma.iloc[t] = sigma

        action = np.argmax(pred_rewards)
        # Selected_Actions[action] = Selected_Actions[action] + 1
        reward, optimal_reward, worst_reward = bandit.get_reward(action, x)
        # optimal_reward = bandit.get_optimal_reward(x)
        agent.update(action, x, reward)

        # To Calculate the MAPE of Prediction in each step
        Vector[6].iloc[t] = np.abs((reward - np.argmax(pred_rewards)) / reward)
        Vector[7].iloc[t] = alpha

        if t == 0:
            Selected_Actions[action].iloc[t] = 1
            Vector[8].iloc[t] = reward
            Vector[9].iloc[t] = optimal_reward
            Vector[10].iloc[t] = worst_reward

            cumulative_rewards[t] = reward
            cumulative_regrets[t] = optimal_reward - reward
        else:
            for i in range(0, n_actions):
                if i == action:
                    Selected_Actions[i].iloc[t] = Selected_Actions[i].iloc[t - 1] + 1
                else:
                    Selected_Actions[i].iloc[t] = Selected_Actions[i].iloc[t - 1]
            cumulative_rewards[t] = cumulative_rewards[t - 1] + reward
            cumulative_regrets[t] = cumulative_regrets[t - 1] + optimal_reward - reward
            Vector[8].iloc[t] = reward
            Vector[9].iloc[t] = optimal_reward
            Vector[10].iloc[t] = worst_reward

    # Number of Features
    Vector[0] = n_features
    # Number of Actions
    Vector[1] = n_actions

    for i in range(0, n_steps):
        # The Highest Probability of Selecting an Arm
        Vector[2].iloc[i] = (Selected_Actions.iloc[i].max() / (i + 1)) / (1 / n_actions)

        # Finding the Difference between the Best and Second Arm
        Vector[3].iloc[i] = ((Selected_Actions.iloc[i].max() - Selected_Actions.iloc[i].nlargest(2).iloc[-1]) / (
                    i + 1)) / (1 / n_actions)

        # Finding the Difference between the Best and Second Miu
        Vector[4].iloc[i] = np.abs((Predicted_Miu.iloc[i].max() - Predicted_Miu.iloc[i].nlargest(2).iloc[-1]) / (
                    Predicted_Miu.iloc[i].max() - Predicted_Miu.iloc[i].min()))

        # Finding the Difference between the Best and Second Miu+Sigma
        Vector[5].iloc[i] = np.abs(((Predicted_Miu.iloc[i] + Predicted_Sigma.iloc[i]).max() -
                                    (Predicted_Miu.iloc[i] + Predicted_Sigma.iloc[i]).nlargest(2).iloc[-1]) / (
                                               (Predicted_Miu.iloc[i] + Predicted_Sigma.iloc[i]).max() - (
                                                   Predicted_Miu.iloc[i] + Predicted_Sigma.iloc[i]).min()))

        Vector[11].iloc[i] = (Vector[8].iloc[i] - Vector[10].iloc[i]) / (Vector[9].iloc[i] - Vector[10].iloc[i])

        Vector[12].iloc[i] = 1 - Vector[11].iloc[i]

    Vector = Vector[[0, 1, 2, 3, 4, 5, 6, 7, 11, 12]]

    Vector.columns = [['Number_of_Features', 'Number_of_Actions', 'High_Probability_of_Selecting',
                       'Difference_Probability', 'Difference_Miu', 'Difference_Miu_Sigma', 'MAPE',
                       'Alpha', 'Reward', 'Regret']]

    Reduced_Vector = pd.DataFrame(np.zeros((Budget, Vector.shape[1])))


    for i in range(0, Budget):

        length = n_steps/Budget

        Reduced_Vector[0].iloc[i] = n_features
        Reduced_Vector[1].iloc[i] = n_actions
        Reduced_Vector[2].iloc[i] = Vector['High_Probability_of_Selecting'].iloc[int(i*length):int((i+1)*length)].mean()
        Reduced_Vector[3].iloc[i] = Vector['Difference_Probability'].iloc[int(i*length):int((i+1)*length)].mean()
        Reduced_Vector[4].iloc[i] = Vector['Difference_Miu'].iloc[int(i*length):int((i+1)*length)].mean()
        Reduced_Vector[5].iloc[i] = Vector['Difference_Miu_Sigma'].iloc[int(i*length):int((i+1)*length)].mean()
        Reduced_Vector[6].iloc[i] = Vector['MAPE'].iloc[int(i*length):int((i+1)*length)].mean()
        Reduced_Vector[7].iloc[i] = Vector['Alpha'].iloc[int(i*length):int((i+1)*length)].mean()
        Reduced_Vector[8].iloc[i] = Vector['Reward'].iloc[int(i*length):int((i+1)*length)].mean()
        Reduced_Vector[9].iloc[i] = Vector['Regret'].iloc[int(i*length):int((i+1)*length)].mean()


    Reduced_Vector.columns = [['Number_of_Features', 'Number_of_Actions', 'High_Probability_of_Selecting',
                                'Difference_Probability', 'Difference_Miu', 'Difference_Miu_Sigma', 'MAPE',
                                'Alpha', 'Reward', 'Regret']]

    return Reduced_Vector


# Define the bandit environment
n_actions = 10
n_features = 10
alpha = 10
Budget = 50
n_steps = 5000

Result = Simulator(n_actions, n_features, alpha, Budget, n_steps)

n_actions = [5, 7]
n_features = [3, 5]
alpha = [1, 10]

# List to store the DataFrames
dfs = []
i = 1
# Iterate over all combinations
for combination in itertools.product(n_actions, n_features, alpha):
    # Unpack the combination
    n_action, n_feature, alpha_value = combination

    # Call your Simulator function
    result_df = Simulator(n_action, n_feature, alpha_value, Budget, n_steps)

    # Adding the parameters as columns to the DataFrame
    result_df['Simulation'] = i
    i = i + 1

    # Append the result to the list
    dfs.append(result_df)

# Concatenate all DataFrames into one
final_df = pd.concat(dfs, ignore_index=True)

# Now final_df contains all your results

# ##### Plot the results
# plt.figure(figsize=(12, 6))
# 
# plt.subplot(121)
# plt.plot(cumulative_rewards)
# plt.xlabel("Steps")
# plt.ylabel("Cumulative Rewards")
# plt.legend()
# 
# plt.subplot(122)
# plt.plot(cumulative_regrets)
# plt.xlabel("Steps")
# plt.ylabel("Cumulative Regrets")
# plt.legend()
# 
# plt.show()
# #####
