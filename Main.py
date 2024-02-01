import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor

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

        return Reward, Optimal_Reward, Worst_Reward



class LinUCB:
    def __init__(self, n_actions, n_features, alpha):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha

        # Initialize parameters
        self.A = np.array([np.identity(n_features) for _ in range(n_actions)])  # action covariance matrix

        self.b = np.array([np.zeros(n_features) for _ in range(n_actions)])  # action reward vector

        self.theta = np.array([np.zeros(n_features) for _ in range(n_actions)])  # action parameter vector

    def update_alpha(self, new_alpha):
        """
        Update the value of alpha.
        :param new_alpha: The new value for alpha.
        """
        self.alpha = new_alpha

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
        Reduced_Vector[7].iloc[i] = alpha
        Reduced_Vector[8].iloc[i] = Vector['Reward'].iloc[int(i*length):int((i+1)*length)].mean()
        Reduced_Vector[9].iloc[i] = Vector['Regret'].iloc[int(i*length):int((i+1)*length)].mean()


    Reduced_Vector.columns = [['Number_of_Features', 'Number_of_Actions', 'High_Probability_of_Selecting',
                                'Difference_Probability', 'Difference_Miu', 'Difference_Miu_Sigma', 'MAPE',
                                'Alpha', 'Reward', 'Regret']]

    return Reduced_Vector


def Data_Selector(Data, n, Budget):

    output = np.zeros((Data['Simulation'].max()[0], Data.shape[1]))
    output = pd.DataFrame(output)

    output.columns = Data.columns
    for i in range(0, len(output)):
        output.iloc[i] = Data.iloc[n+(i*Budget)]

    Regret = output['Regret']
    output.drop(['Reward', 'Regret', 'Simulation'], axis = 1, inplace= True)

    return output, Regret


def HPO_Transfer(X, Y, input):
    # Fit a Gaussian Mixture Model to your data

    gaussian_process = GaussianProcessRegressor()
    gaussian_process.fit(X, Y)


    input = input[:X.shape[1]]

    results = np.zeros((20, 2))
    results = pd.DataFrame(results)
    for i in range(0, 20):
        alpha = 10 / (i+1)
        input[7] = alpha
        mean_prediction, std_prediction = gaussian_process.predict(np.array(input).reshape(1, -1), return_std=True)
        results[0].iloc[i] = mean_prediction
        results[1].iloc[i] = alpha

    Optimum_Alpha = results[1].iloc[np.argmin(results[0])]

    return Optimum_Alpha


def HPO_Sorrugate(Alpha_Vector, Result_Vector):
    # Fit a Gaussian Mixture Model to your data

    gaussian_process = GaussianProcessRegressor()
    Alpha_Vector = pd.DataFrame(Alpha_Vector)
    gaussian_process.fit(Alpha_Vector, Result_Vector)



    results = np.zeros((20, 2))
    results = pd.DataFrame(results)
    for i in range(0, 20):
        alpha = 10 / (i+1)
        pd.DataFrame([alpha])
        mean_prediction, std_prediction = gaussian_process.predict(pd.DataFrame([alpha]), return_std=True)
        results[0].iloc[i] = mean_prediction
        results[1].iloc[i] = alpha

    Optimum_Alpha = results[1].iloc[np.argmin(results[0])]

    return Optimum_Alpha


def Model_Selection(X, Y, Previous_Alpha, Alpha_Vector, Result_Vector, Previous_Input, Previous_Result):
    gaussian_process = GaussianProcessRegressor()
    gaussian_process.fit(X, Y)
    input = Previous_Input[:X.shape[1]]
    input[7] = Previous_Alpha
    mean_prediction, std_prediction = gaussian_process.predict(np.array(input).reshape(1, -1), return_std=True)

    gaussian_process_alpha = GaussianProcessRegressor()
    gaussian_process_alpha.fit(pd.DataFrame(Alpha_Vector), Result_Vector)
    mean_prediction_alpha, std_prediction_alpha = gaussian_process_alpha.predict(pd.DataFrame([Previous_Alpha]), return_std=True)

    Transfer_Mape = np.abs(Previous_Result - mean_prediction) / Previous_Result
    Sorrugate_Mape = np.abs(Previous_Result - mean_prediction_alpha) / Previous_Result

    if Transfer_Mape >= Sorrugate_Mape:
        Selected_model = 'Sorrugate'
    else:
        Selected_model = 'Transfer'

    return Selected_model



def Target_Task_Optimization(n_actions, n_features,initial_alpha, Total_steps, Budget, final_df):

    bandit = Bandit(n_actions, n_features)

    linucb_agent = LinUCB(n_actions, n_features, initial_alpha)


    agents = [linucb_agent]

    n_steps = int(Total_steps / Budget)

    cumulative_rewards = np.zeros(Total_steps)
    cumulative_regrets = np.zeros(Total_steps)

    agent = linucb_agent

    Selected_Actions = np.zeros((Total_steps, n_actions))
    Selected_Actions = pd.DataFrame(Selected_Actions)

    Predicted_Rewards = np.zeros((Total_steps, n_actions))
    Predicted_Rewards = pd.DataFrame(Predicted_Rewards)

    Predicted_Miu = np.zeros((Total_steps, n_actions))
    Predicted_Miu = pd.DataFrame(Predicted_Miu)

    Predicted_Sigma = np.zeros((Total_steps, n_actions))
    Predicted_Sigma = pd.DataFrame(Predicted_Sigma)

    Vector = np.zeros((Total_steps, 13))
    Vector = pd.DataFrame(Vector)
    Reduced_Vector = pd.DataFrame(np.zeros((Budget, Vector.shape[1]+1)))

    Alpha_Vector = []

    for z in range(0, Budget):
        if z < 2:

            X, Y = Data_Selector(final_df, z, Budget)
            alpha = X['Alpha'].iloc[np.argmin(Y)][0]
            linucb_agent.update_alpha(alpha)
            Selected_Model = 'Nazdiki'

        if z >= 2:
            Selected_Model = Model_Selection(X_Previous_Step, Y_Previous_Step, Previous_Alpha, Alpha_Vector[:-1], Reduced_Vector[9].iloc[:z-1], Reduced_Vector.iloc[z-2], Reduced_Vector[9].iloc[z-1])
            if Selected_Model == 'Sorrugate':
                X, Y = Data_Selector(final_df, z, Budget)
                alpha = HPO_Sorrugate(Alpha_Vector, Reduced_Vector[9].iloc[:z])
                linucb_agent.update_alpha(alpha)

            else:
                X, Y = Data_Selector(final_df, z, Budget)
                alpha = HPO_Transfer(X, Y, Reduced_Vector.iloc[z-1])
                linucb_agent.update_alpha(alpha)

        X_Previous_Step = X
        Y_Previous_Step = Y
        Previous_Alpha = alpha
        Alpha_Vector.append(alpha)


        for o in range(0, n_steps):

            # if t >= 2*(n_steps/Budget):
            #     if t % (n_steps/Budget) == 0:
            #         alpha = HPO()
            #         linucb_agent.update_alpha(alpha)
            # else:
            #     alpha = initial_alpha

            t = o + z * n_steps

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


            # The Highest Probability of Selecting an Arm
            Vector[2].iloc[t] = (Selected_Actions.iloc[t].max() / (t + 1)) / (1 / n_actions)

            # Finding the Difference between the Best and Second Arm
            Vector[3].iloc[t] = ((Selected_Actions.iloc[t].max() - Selected_Actions.iloc[t].nlargest(2).iloc[-1]) / (
                        t + 1)) / (1 / n_actions)

            # Finding the Difference between the Best and Second Miu
            Vector[4].iloc[t] = np.abs((Predicted_Miu.iloc[t].max() - Predicted_Miu.iloc[t].nlargest(2).iloc[-1]) / (
                        Predicted_Miu.iloc[t].max() - Predicted_Miu.iloc[t].min()))

            # Finding the Difference between the Best and Second Miu+Sigma
            Vector[5].iloc[t] = np.abs(((Predicted_Miu.iloc[t] + Predicted_Sigma.iloc[t]).max() -
                                        (Predicted_Miu.iloc[t] + Predicted_Sigma.iloc[t]).nlargest(2).iloc[-1]) / (
                                                (Predicted_Miu.iloc[t] + Predicted_Sigma.iloc[t]).max() - (
                                                    Predicted_Miu.iloc[t] + Predicted_Sigma.iloc[t]).min()))

            Vector[11].iloc[t] = (Vector[8].iloc[t] - Vector[10].iloc[t]) / (Vector[9].iloc[t] - Vector[10].iloc[t])

            Vector[12].iloc[t] = 1 - Vector[11].iloc[t]


        # Number of Features
        Vector[0] = n_features
        # Number of Actions
        Vector[1] = n_actions

        length = n_steps

        Reduced_Vector[0].iloc[z] = n_features
        Reduced_Vector[1].iloc[z] = n_actions
        Reduced_Vector[2].iloc[z] = Vector[2].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[3].iloc[z] = Vector[3].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[4].iloc[z] = Vector[4].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[5].iloc[z] = Vector[5].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[6].iloc[z] = Vector[6].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[8].iloc[z] = Vector[11].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[9].iloc[z] = Vector[12].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[10].iloc[z] = z
        Reduced_Vector[11].iloc[z] = alpha
        Reduced_Vector[12].iloc[z] = Vector[7].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[13].iloc[z] = Selected_Model
    Reduced_Vector[7] = 0
    Reduced_Vector[7] = Reduced_Vector[12]

    return Reduced_Vector, Vector


# Define the bandit environment

Budget = 50
n_steps = 5000
n_actions = [5, 7, 10, 12, 15, 18, 20]
n_features = [3, 5, 7, 10, 12, 15, 18, 20]
alpha = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 5]

# List to store the DataFrames
dfs = []
i = 1
# Iterate over all combinations
for combination in itertools.product(n_actions, n_features, alpha):
    print(i)
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

initial_alpha = 1
Total_steps = 5000
n_actions = 10
n_features = 10

Reduced_Vector, Vector = Target_Task_Optimization(n_actions, n_features,initial_alpha, Total_steps, Budget, final_df)

