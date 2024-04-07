import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle

pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

# The main Bandit class
class Bandit:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.theta = np.random.randn(n_actions, n_features)

    def update_theta(self, new_theta):
        """
        Update the value of alpha.
        :param new_alpha: The new value for alpha.
        """
        self.theta = new_theta

    def get_reward(self, action, x):

        randomness = np.random.normal()
        Reward = (x @ self.theta[action] + randomness)
        Optimal_Reward = np.max(x @ self.theta.T + randomness)
        Worst_Reward = np.min(x @ self.theta.T + randomness)

        # Reward = (Reward - Worst_Reward) / (Optimal_Reward - Worst_Reward)
        # Optimal_Reward = 1

        return Reward, Optimal_Reward, Worst_Reward

# The neural network object
class NeuralNetwork(nn.Module):
    def __init__(self, n_features):
        super(NeuralNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layer(x)

# The neural bandit class updates neural network weights and predict the rewards
class NeuralBandit:
    def __init__(self, n_actions, n_features, learning_rate, momentum):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Initialize the neural network model for each action
        self.models = [NeuralNetwork(n_features) for _ in range(n_actions)]
        self.optimizers = [
            optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
            for model in self.models
        ]
        self.criterion = nn.MSELoss()

    def predict(self, context):
        context_tensor = torch.tensor(context, dtype=torch.float32)  # Convert to tensor
        with torch.no_grad():
            return torch.cat(
                [model(context_tensor).reshape(1) for model in self.models]
            )

    def update(self, action, context, reward):
        self.optimizers[action].zero_grad()
        context_tensor = torch.tensor(context, dtype=torch.float32)  # Convert to tensor
        reward_tensor = torch.tensor(reward, dtype=torch.float32)  # Convert to tensor
        pred_reward = self.models[action](context_tensor)
        loss = self.criterion(pred_reward, reward_tensor)
        loss.backward()
        # Apply gradient clipping before optimizer step
        torch.nn.utils.clip_grad_norm_(self.models[action].parameters(), max_norm=1)
        self.optimizers[action].step()

    def update_alpha(self, new_learning_rate, new_momentum):
        """
        Update the value of alpha.
        :param new_alpha: The new value for alpha.
        """
        self.learning_rate = new_learning_rate
        self.momentum = new_momentum

# The simulation function which generates data for transfer learning
def Simulator(n_actions, n_features, learning_rate, momentum, Budget, n_steps, Non_Stationary_Step, Non_Stationary ):
    """
    :param n_actions: Number of actions
    :param n_features: Number of features
    :param learning_rate: learning rate of neural network
    :param momentum: momentum of the neural network
    :param Budget: Number of steps in which data are gathered and summarized
    :param n_steps: Available budget for bandit's run
    :param Non_Stationary_Step: Number of step in which reward function is changed
    :param Non_Stationary: Stationary status - True means there is changes in reward function
    :return: A dataframe which is a knowledge based for transfer learning
    """
    bandit = Bandit(n_actions, n_features)
    neural_agent = NeuralBandit(n_actions, n_features, learning_rate, momentum)

    agents = [neural_agent]

    cumulative_rewards = np.zeros(n_steps)
    cumulative_regrets = np.zeros(n_steps)

    agent = neural_agent

    Selected_Actions = np.zeros((n_steps, n_actions))
    Selected_Actions = pd.DataFrame(Selected_Actions)

    Predicted_Rewards = np.zeros((n_steps, n_actions))
    Predicted_Rewards = pd.DataFrame(Predicted_Rewards)


    Vector = np.zeros((n_steps, 13))
    Vector = pd.DataFrame(Vector)

    for t in range(n_steps):


        if (Non_Stationary == True) and  (t == (n_steps/Budget) * Non_Stationary_Step):
            new_coef = shuffle(bandit.theta)

            bandit.update_theta(new_coef)

        learning_rate = learning_rate
        x = np.random.randn(n_features)
        pred_rewards = agent.predict([x])
        Predicted_Rewards.iloc[t] = pred_rewards.numpy()

        action = np.argmax(pred_rewards.numpy())
        # Selected_Actions[action] = Selected_Actions[action] + 1
        reward, optimal_reward, worst_reward = bandit.get_reward(action, x)
        # optimal_reward = bandit.get_optimal_reward(x)
        agent.update(action, x, reward)

        # To Calculate the MAPE of Prediction in each step
        Vector[6].iloc[t] = np.abs((reward - np.argmax(pred_rewards.numpy())) / reward)
        Vector[7].iloc[t] = learning_rate

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


        Vector[11].iloc[i] = (Vector[8].iloc[i] - Vector[10].iloc[i]) / (Vector[9].iloc[i] - Vector[10].iloc[i])

        Vector[12].iloc[i] = 1 - Vector[11].iloc[i]

    Vector = Vector[[0, 1, 2, 3, 4, 5, 6, 7, 11, 12]]

    Vector.columns = [['Number_of_Features', 'Number_of_Actions', 'High_Probability_of_Selecting',
                       'Difference_Probability', 'Difference_Miu', 'Difference_Miu_Sigma', 'MAPE',
                       'learning_rate', 'Reward', 'Regret']]

    Reduced_Vector = pd.DataFrame(np.zeros((Budget, Vector.shape[1])))


    for i in range(0, Budget):

        length = n_steps/Budget

        Reduced_Vector[0].iloc[i] = n_features
        Reduced_Vector[1].iloc[i] = n_actions
        Reduced_Vector[2].iloc[i] = Vector['High_Probability_of_Selecting'].iloc[int(i*length):int((i+1)*length)].mean()
        Reduced_Vector[3].iloc[i] = Vector['Difference_Probability'].iloc[int(i*length):int((i+1)*length)].mean()
        Reduced_Vector[6].iloc[i] = Vector['MAPE'].iloc[int(i*length):int((i+1)*length)].mean()
        Reduced_Vector[7].iloc[i] = learning_rate
        Reduced_Vector[5].iloc[i] = momentum
        Reduced_Vector[8].iloc[i] = Vector['Reward'].iloc[int(i*length):int((i+1)*length)].mean()
        Reduced_Vector[9].iloc[i] = Vector['Regret'].iloc[int(i*length):int((i+1)*length)].mean()


    Reduced_Vector.columns = [['Number_of_Features', 'Number_of_Actions', 'High_Probability_of_Selecting',
                                'Difference_Probability', 'Difference_Miu', 'Momentum', 'MAPE',
                                'learning_rate', 'Reward', 'Regret']]

    return Reduced_Vector

## Combinations of different parameters to generate a knowledge-based for transfer learning by simulation

# Number of steps in which data are gathered and summarized
Budget = 50
# Available budget for bandit's run
n_steps = 5000
# Number of actions
n_actions = [5, 7, 10, 12, 15]
# Number of features
n_features = [3, 5, 7, 10, 12, 15]
# Learning Rate
learning_rates = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
# Momentum
momentums = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Number of replications in simulation
replication = [1, 2, 3]
# Stationary status- if it is true reward function is changed in some steps
Non_Stationary = True
Non_Stationary_Step = 25

# List to store the DataFrames
dfs = []
i = 1
# Iterate over all combinations
for combination in itertools.product(n_actions, n_features, learning_rates, momentums, replication):
    print(i)
    # Unpack the combination
    n_action, n_feature, learning_rate, momentum, replication = combination

    # Call your Simulator function
    result_df = Simulator(n_action, n_feature, learning_rate, momentum, Budget, n_steps, Non_Stationary_Step, Non_Stationary)

    # Adding the parameters as columns to the DataFrame
    result_df['Simulation'] = i
    result_df['Replication'] = replication
    i = i + 1

    # Append the result to the list
    dfs.append(result_df)

# Concatenate all DataFrames into one
# final_df = pd.concat(dfs, ignore_index=True)
# final_df.to_excel("NN_Final_Shuffle.xlsx")

def Data_Selector(Data, n, Budget):
    """
    :param Data: Simulation data set
    :param n: current step
    :param Budget: Total steps
    :return: Features dataset and Regret column
    """
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
    X.columns = pd.RangeIndex(X.columns.size)

    gaussian_process.fit(X, Y)


    input = input[:X.shape[1]]



    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
    momentums = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    comb = len(learning_rates) * len(momentums)

    results = np.zeros((comb, 3))
    results = pd.DataFrame(results)

    i = 0
    for combination in itertools.product(learning_rates, momentums):
        learning_rate, momentum = combination
        input[7] = learning_rate
        input[5] = momentum
        mean_prediction, std_prediction = gaussian_process.predict(np.array(input).reshape(1, -1), return_std=True)
        results[0].iloc[i] = mean_prediction - std_prediction
        results[1].iloc[i] = learning_rate
        results[2].iloc[i] = momentum

        i += 1
    Optimum_Alpha = [results[1].iloc[np.argmin(results[0])], results[2].iloc[np.argmin(results[0])] ]

    return Optimum_Alpha


def HPO_Sorrugate(Alpha_Vector, Result_Vector):
    # Fit a Gaussian Mixture Model to your data

    gaussian_process = GaussianProcessRegressor()
    Alpha_Vector = pd.DataFrame(Alpha_Vector)
    gaussian_process.fit(Alpha_Vector[1:], Result_Vector[1:])

    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
    momentums = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    comb = len(learning_rates) * len(momentums)
    results = np.zeros((comb, 3))
    results = pd.DataFrame(results)

    i= 0
    for combination in itertools.product(learning_rates, momentums):
        learning_rate, momentum = combination

        mean_prediction, std_prediction = gaussian_process.predict(pd.DataFrame([learning_rate, momentum]).T, return_std=True)
        results[0].iloc[i] = mean_prediction - std_prediction
        results[1].iloc[i] = learning_rate
        results[2].iloc[i] = momentum

        i += 1
    Optimum_Alpha = [results[1].iloc[np.argmin(results[0])], results[2].iloc[np.argmin(results[0])] ]

    return Optimum_Alpha


def Model_Selection(X, Y, Previous_Learningrate, Previous_momentum, Alpha_Vector, Result_Vector, Previous_Input, Previous_Result):
    gaussian_process = GaussianProcessRegressor()
    X.columns = pd.RangeIndex(X.columns.size)
    gaussian_process.fit(X, Y)
    input = Previous_Input[:X.shape[1]]
    input[7] = Previous_Learningrate
    input[5] = Previous_momentum

    mean_prediction, std_prediction = gaussian_process.predict(np.array(input).reshape(1, -1), return_std=True)

    gaussian_process_alpha = GaussianProcessRegressor()
    gaussian_process_alpha.fit(pd.DataFrame(Alpha_Vector), Result_Vector)
    mean_prediction_alpha, std_prediction_alpha = gaussian_process_alpha.predict(pd.DataFrame([Previous_Learningrate, Previous_momentum]).T, return_std=True)

    Transfer_Mape = np.abs(Previous_Result - mean_prediction) / Previous_Result
    Sorrugate_Mape = np.abs(Previous_Result - mean_prediction_alpha) / Previous_Result

    if 1.2 * Transfer_Mape >= Sorrugate_Mape:
        Selected_model = 'Sorrugate'
    else:
        Selected_model = 'Transfer'

    return Selected_model



def Target_Task_Optimization(n_actions, n_features, bandit,initial_alpha, Total_steps, Budget, final_df, Non_Stationary_Step, Non_Stationary,New_theta):



    nueral_agent = NeuralBandit(n_actions, n_features, learning_rate, momentum)

    agent = nueral_agent
    n_steps = int(Total_steps / Budget)

    cumulative_rewards = np.zeros(Total_steps)
    cumulative_regrets = np.zeros(Total_steps)


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
            # alpha = X['Alpha'].iloc[np.argmin(Y)][0]
            new_learning_rate = 0.01
            new_momentum = 0.5
            agent.update_alpha(new_learning_rate, new_momentum)
            Selected_Model = 'Nazdiki'

        if z >= 2:
            Selected_Model = Model_Selection(X_Previous_Step, Y_Previous_Step, Previous_Learningrate, Previous_momentum, Alpha_Vector[:-1], Reduced_Vector[9].iloc[:z-1], Reduced_Vector.iloc[z-2], Reduced_Vector[9].iloc[z-1])
            if Selected_Model == 'Sorrugate':
                X, Y = Data_Selector(final_df, z, Budget)
                [new_learning_rate ,new_momentum] = HPO_Sorrugate(Alpha_Vector, Reduced_Vector[9].iloc[:z])
                agent.update_alpha(new_learning_rate, new_momentum)

            else:
                X, Y = Data_Selector(final_df, z, Budget)
                [new_learning_rate ,new_momentum] = HPO_Transfer(X, Y, Reduced_Vector.iloc[z-1])
                agent.update_alpha(new_learning_rate, new_momentum)

        X_Previous_Step = X
        Y_Previous_Step = Y
        Previous_Learningrate = new_learning_rate
        Previous_momentum = new_momentum
        Alpha_Vector.append([new_learning_rate, new_momentum])

        if(z==Non_Stationary_Step and Non_Stationary == True):
            bandit.theta = New_theta
            bandit.update_theta(bandit.theta)

        for o in range(0, n_steps):

            # if t >= 2*(n_steps/Budget):
            #     if t % (n_steps/Budget) == 0:
            #         alpha = HPO()
            #         linucb_agent.update_alpha(alpha)
            # else:
            #     alpha = initial_alpha

            t = o + z * n_steps

            x = np.random.randn(n_features)
            pred_rewards = agent.predict([x]).numpy()
            Predicted_Rewards.iloc[t] = pred_rewards


            action = np.argmax(pred_rewards)
            # Selected_Actions[action] = Selected_Actions[action] + 1
            reward, optimal_reward, worst_reward = bandit.get_reward(action, x)
            # optimal_reward = bandit.get_optimal_reward(x)
            agent.update(action, x, reward)

            # To Calculate the MAPE of Prediction in each step

            Vector[6].iloc[t] = np.abs((reward - np.argmax(pred_rewards)) / reward)
            Vector[7].iloc[t] = new_learning_rate
            Vector[5].iloc[t] = new_momentum

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



            Vector[11].iloc[t] = (Vector[8].iloc[t] - Vector[10].iloc[t]) / (Vector[9].iloc[t] - Vector[10].iloc[t])

            Vector[12].iloc[t] = 1 - Vector[11].iloc[t]
            # print(pred_rewards)
            # print(agent.learning_rate)
            # print(agent.momentum)


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
        Reduced_Vector[11].iloc[z] = new_learning_rate
        Reduced_Vector[12].iloc[z] = Vector[7].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[13].iloc[z] = Selected_Model
        Reduced_Vector[7] = Reduced_Vector[11]
    Reduced_Vector[7] = 0
    Reduced_Vector[7] = Reduced_Vector[12]

    return Reduced_Vector, Vector, bandit



initial_alpha = 1
Total_steps = 5000
n_actions = 10
n_features = 10
learning_rate = 0.1
momentum = 0.7
Budget = 50
n_steps = 5000
Non_Stationary = True
Non_Stationary_Step = 25

Main_Bandit = Bandit(n_actions, n_features)

bandit = Main_Bandit
Main_theta = Main_Bandit.theta
bandit.theta = Main_theta
# New_theta = - Main_theta
# New_theta = Main_theta + np.random.random()*5
New_theta = shuffle(Main_theta)
Reduced_Vector_Algorithm, Vector_Algorithm, bandit_new = Target_Task_Optimization(n_actions, n_features, bandit, initial_alpha, Total_steps, Budget, final_df, Non_Stationary_Step, Non_Stationary, New_theta)


def Bayesian_Task_Optimization(n_actions, n_features, bandit, initial_alpha, Total_steps, Budget, final_df, Non_Stationary_Step, Non_Stationary, New_theta):
    learning_rate = 0.1
    momentum = 0.9
    nueral_agent = NeuralBandit(n_actions, n_features, learning_rate, momentum)

    def Acquisition(Alpha_Vector, Result_Vector):
        # Fit a Gaussian Mixture Model to your data
        gaussian_process = GaussianProcessRegressor()
        Alpha_Vector = pd.DataFrame(Alpha_Vector)
        gaussian_process.fit(Alpha_Vector, Result_Vector)

        results = np.zeros((100, 3))
        results = pd.DataFrame(results)

        learning_rates = [0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25]
        momentums = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        i = 0
        for combination in itertools.product(learning_rates, momentums):
            learning_rate, momentum = combination

            mean_prediction, std_prediction = gaussian_process.predict(pd.DataFrame([learning_rate, momentum]).T, return_std=True)
            results[0].iloc[i] = mean_prediction - std_prediction
            results[1].iloc[i] = learning_rate
            results[2].iloc[i] = momentum

            i += 1
        Optimum_Alpha = [results[1].iloc[np.argmin(results[0])], results[2].iloc[np.argmin(results[0])]]

        return Optimum_Alpha


    agent = nueral_agent

    n_steps = int(Total_steps / Budget)

    cumulative_rewards = np.zeros(Total_steps)
    cumulative_regrets = np.zeros(Total_steps)


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
        if z == 0:

            learning_rate = np.random.uniform(0.001, 0.3)
            momentum = np.random.uniform(0.1, 0.9)

            agent.update_alpha(learning_rate, momentum)

        if z >= 1:

            [learning_rate, momentum] = Acquisition(Alpha_Vector, Reduced_Vector[9].iloc[:z])
            agent.update_alpha(learning_rate, momentum)


        Alpha_Vector.append([learning_rate, momentum])

        if(z==Non_Stationary_Step and Non_Stationary == True):
            bandit.theta = New_theta
            bandit.update_theta(bandit.theta)

        for o in range(0, n_steps):

            # if t >= 2*(n_steps/Budget):
            #     if t % (n_steps/Budget) == 0:
            #         alpha = HPO()
            #         linucb_agent.update_alpha(alpha)
            # else:
            #     alpha = initial_alpha

            t = o + z * n_steps

            x = np.random.randn(n_features)
            pred_rewards = agent.predict([x]).numpy()
            Predicted_Rewards.iloc[t] = pred_rewards


            action = np.argmax(pred_rewards)
            # Selected_Actions[action] = Selected_Actions[action] + 1
            reward, optimal_reward, worst_reward = bandit.get_reward(action, x)
            # optimal_reward = bandit.get_optimal_reward(x)
            agent.update(action, x, reward)

            # To Calculate the MAPE of Prediction in each step

            Vector[6].iloc[t] = np.abs((reward - np.argmax(pred_rewards)) / reward)
            Vector[7].iloc[t] = learning_rate
            Vector[5].iloc[t] = momentum


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
        Reduced_Vector[4].iloc[z] = 0
        Reduced_Vector[5].iloc[z] = momentum
        Reduced_Vector[6].iloc[z] = Vector[6].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[8].iloc[z] = Vector[11].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[9].iloc[z] = Vector[12].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[10].iloc[z] = z
        Reduced_Vector[11].iloc[z] = learning_rate
        Reduced_Vector[12].iloc[z] = Vector[7].iloc[int(z*length):int((z+1)*length)].mean()
    Reduced_Vector[7] = 0
    Reduced_Vector[7] = Reduced_Vector[12]

    return Reduced_Vector, Vector, bandit


bandit = Bandit(n_actions, n_features)
bandit.theta = Main_theta
Reduced_Vector_Bayesian, Vector_Bayesian, bandit_Bayes = Bayesian_Task_Optimization(n_actions, n_features, bandit, initial_alpha, Total_steps, Budget, final_df, Non_Stationary_Step, Non_Stationary, New_theta)

def RandomSearch_Task_Optimization(n_actions, n_features, bandit, initial_alpha, Total_steps, Budget, final_df, Non_Stationary_Step, Non_Stationary, New_theta):

    learning_rate = 0.1
    momentum = 0.9
    nueral_agent = NeuralBandit(n_actions, n_features, learning_rate, momentum)

    n_steps = int(Total_steps / Budget)

    cumulative_rewards = np.zeros(Total_steps)
    cumulative_regrets = np.zeros(Total_steps)

    agent = nueral_agent

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
    min_func_val = np.inf

    for z in range(0, Budget):
        epsilon = 0.5
        if z == 0:
            learning_rate = np.random.uniform(0.0001, 0.9)
            momentum = np.random.uniform(0.1, 0.9)

            agent.update_alpha(learning_rate, momentum)
            best_alpha = [learning_rate, momentum]

        if z >= 1:

            if Reduced_Vector[9].iloc[z - 1] < min_func_val:
                min_func_val = Reduced_Vector[9].iloc[z - 1]
                best_alpha = [learning_rate, momentum]
            if np.random.random() <= epsilon:
                learning_rate = np.random.uniform(0.001, 0.99)
                momentum = np.random.uniform(0.1, 0.99)
            else:
                alpha = best_alpha

            agent.update_alpha(learning_rate, momentum)

        Alpha_Vector.append([learning_rate, momentum])

        if(z==Non_Stationary_Step and Non_Stationary == True):
            bandit.theta = New_theta
            bandit.update_theta(bandit.theta)

        for o in range(0, n_steps):

            # if t >= 2*(n_steps/Budget):
            #     if t % (n_steps/Budget) == 0:
            #         alpha = HPO()
            #         linucb_agent.update_alpha(alpha)
            # else:
            #     alpha = initial_alpha

            t = o + z * n_steps

            x = np.random.randn(n_features)
            pred_rewards = agent.predict([x]).numpy()
            Predicted_Rewards.iloc[t] = pred_rewards


            action = np.argmax(pred_rewards)
            # Selected_Actions[action] = Selected_Actions[action] + 1
            reward, optimal_reward, worst_reward = bandit.get_reward(action, x)
            # optimal_reward = bandit.get_optimal_reward(x)
            agent.update(action, x, reward)

            # To Calculate the MAPE of Prediction in each step

            Vector[6].iloc[t] = np.abs((reward - np.argmax(pred_rewards)) / reward)
            Vector[7].iloc[t] = learning_rate
            Vector[5].iloc[t] = momentum


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
        Reduced_Vector[4].iloc[z] = 0
        Reduced_Vector[5].iloc[z] = momentum
        Reduced_Vector[6].iloc[z] = Vector[6].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[8].iloc[z] = Vector[11].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[9].iloc[z] = Vector[12].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[10].iloc[z] = z
        Reduced_Vector[11].iloc[z] = learning_rate
        Reduced_Vector[12].iloc[z] = Vector[7].iloc[int(z*length):int((z+1)*length)].mean()
    Reduced_Vector[7] = 0
    Reduced_Vector[7] = Reduced_Vector[12]

    return Reduced_Vector, Vector, bandit


bandit = Bandit(n_actions, n_features)
bandit.theta = Main_theta
Reduced_Vector_RandomSearch, Vector_RandomSearch, bandit_Random = RandomSearch_Task_Optimization(n_actions, n_features, bandit, initial_alpha, Total_steps, Budget, final_df, Non_Stationary_Step, Non_Stationary, New_theta)


def GradualDecresement_Task_Optimization(n_actions, n_features, bandit, initial_alpha, Total_steps, Budget, final_df, Non_Stationary_Step, Non_Stationary, New_theta):

    learning_rate = 0.1
    momentum = 0.9
    nueral_agent = NeuralBandit(n_actions, n_features, learning_rate, momentum)



    n_steps = int(Total_steps / Budget)

    cumulative_rewards = np.zeros(Total_steps)
    cumulative_regrets = np.zeros(Total_steps)

    agent = nueral_agent

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
    min_func_val = np.inf

    for z in range(0, Budget):

        learning_rate = 0.9 / (z+1)
        momentum = 0.1 + (0.09 / (Budget-z))
        agent.update_alpha(learning_rate, momentum)

        Alpha_Vector.append([learning_rate, momentum])

        if(z==Non_Stationary_Step and Non_Stationary == True):
            bandit.theta = New_theta
            bandit.update_theta(bandit.theta)

        for o in range(0, n_steps):

            # if t >= 2*(n_steps/Budget):
            #     if t % (n_steps/Budget) == 0:
            #         alpha = HPO()
            #         linucb_agent.update_alpha(alpha)
            # else:
            #     alpha = initial_alpha

            t = o + z * n_steps

            x = np.random.randn(n_features)
            pred_rewards = agent.predict([x]).numpy()
            Predicted_Rewards.iloc[t] = pred_rewards


            action = np.argmax(pred_rewards)
            # Selected_Actions[action] = Selected_Actions[action] + 1
            reward, optimal_reward, worst_reward = bandit.get_reward(action, x)
            # optimal_reward = bandit.get_optimal_reward(x)
            agent.update(action, x, reward)

            # To Calculate the MAPE of Prediction in each step

            Vector[6].iloc[t] = np.abs((reward - np.argmax(pred_rewards)) / reward)
            Vector[7].iloc[t] = learning_rate
            Vector[5].iloc[t] = momentum

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
        Reduced_Vector[4].iloc[z] = 0
        Reduced_Vector[5].iloc[z] = momentum
        Reduced_Vector[6].iloc[z] = Vector[6].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[8].iloc[z] = Vector[11].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[9].iloc[z] = Vector[12].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[10].iloc[z] = z
        Reduced_Vector[11].iloc[z] = learning_rate
        Reduced_Vector[12].iloc[z] = Vector[7].iloc[int(z*length):int((z+1)*length)].mean()
    Reduced_Vector[7] = 0
    Reduced_Vector[7] = Reduced_Vector[12]

    return Reduced_Vector, Vector, bandit

bandit = Bandit(n_actions, n_features)
bandit.theta = Main_theta
Reduced_Vector_GradualDecresement, Vector_GradualDecresement, bandit_Dec = GradualDecresement_Task_Optimization(n_actions, n_features, bandit, initial_alpha, Total_steps, Budget, final_df, Non_Stationary_Step, Non_Stationary, New_theta)



def BOBSoftmax_Task_Optimization(n_actions, n_features, bandit, initial_alpha, Total_steps, Budget, final_df, Non_Stationary_Step, Non_Stationary, New_theta):

    learning_rate = 0.1
    momentum = 0.9
    nueral_agent = NeuralBandit(n_actions, n_features, learning_rate, momentum)

    n_steps = int(Total_steps / Budget)

    cumulative_rewards = np.zeros(Total_steps)
    cumulative_regrets = np.zeros(Total_steps)

    agent = nueral_agent

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
    min_func_val = np.inf

    learning_rates = [0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25]
    momentums = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    comb = len(learning_rates) * len(momentums)

    selected_alpha = np.zeros((comb,4))
    selected_alpha = pd.DataFrame(selected_alpha)

    for z in range(0, Budget):
        if z == 0:
            index1 = random.randrange(len(learning_rates))
            index2 = random.randrange(len(momentums))

            learning_rate = learning_rates[index1]
            momentum = momentums[index2]

            selected_alpha[0].iloc[index1 + 10*index2] = selected_alpha[0].iloc[index1 + 10*index2] + 1
            agent.update_alpha(learning_rate, momentum)

        if z >= 1:

            selected_alpha[1].iloc[index1 + 10*index2] = selected_alpha[1].iloc[index1 + 10*index2] + (1 - Reduced_Vector[9].iloc[z - 1])
            for l in range(0, len(selected_alpha)):
                if selected_alpha[0].iloc[l] == 0:
                    selected_alpha[2].iloc[l] = 0.7
                else:
                    selected_alpha[2].iloc[l] = selected_alpha[1].iloc[l] / selected_alpha[0].iloc[l]

            selected_alpha[3] = np.exp(selected_alpha[2])
            selected_alpha[3] = selected_alpha[3] / selected_alpha[3].sum()
            andis = np.random.choice(comb, 1, p=list(selected_alpha[3]))[0]
            index1 = np.mod(andis, 10)
            index2 = int(np.floor(andis/10))

            learning_rate = learning_rates[index1]
            momentum = momentums[index2]

            selected_alpha[0].iloc[index1 + 10*index2] = selected_alpha[0].iloc[index1 + 10*index2] + 1
            agent.update_alpha(learning_rate, momentum)


        if(z==Non_Stationary_Step and Non_Stationary == True):
            bandit.theta = New_theta
            bandit.update_theta(bandit.theta)

        for o in range(0, n_steps):

            # if t >= 2*(n_steps/Budget):
            #     if t % (n_steps/Budget) == 0:
            #         alpha = HPO()
            #         linucb_agent.update_alpha(alpha)
            # else:
            #     alpha = initial_alpha

            t = o + z * n_steps

            x = np.random.randn(n_features)
            pred_rewards = agent.predict([x]).numpy()
            Predicted_Rewards.iloc[t] = pred_rewards


            action = np.argmax(pred_rewards)
            # Selected_Actions[action] = Selected_Actions[action] + 1
            reward, optimal_reward, worst_reward = bandit.get_reward(action, x)
            # optimal_reward = bandit.get_optimal_reward(x)
            agent.update(action, x, reward)

            # To Calculate the MAPE of Prediction in each step

            Vector[6].iloc[t] = np.abs((reward - np.argmax(pred_rewards)) / reward)
            Vector[7].iloc[t] = learning_rate
            Vector[5].iloc[t] = momentum

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
        Reduced_Vector[4].iloc[z] = 0
        Reduced_Vector[5].iloc[z] = momentum
        Reduced_Vector[6].iloc[z] = Vector[6].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[8].iloc[z] = Vector[11].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[9].iloc[z] = Vector[12].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[10].iloc[z] = z
        Reduced_Vector[11].iloc[z] = learning_rate
        Reduced_Vector[12].iloc[z] = Vector[7].iloc[int(z*length):int((z+1)*length)].mean()
    Reduced_Vector[7] = 0
    Reduced_Vector[7] = Reduced_Vector[12]

    return Reduced_Vector, Vector

bandit = Bandit(n_actions, n_features)
bandit.theta = Main_theta
Reduced_Vector_BOBSoftmax, Vector_BOBSoftmax = BOBSoftmax_Task_Optimization(n_actions, n_features, bandit, initial_alpha, Total_steps, Budget, final_df, Non_Stationary_Step, Non_Stationary, New_theta)


def BOBUCB_Task_Optimization(n_actions, n_features, bandit, initial_alpha, Total_steps, Budget, final_df, Non_Stationary_Step, Non_Stationary, New_theta):

    learning_rate = 0.1
    momentum = 0.9
    nueral_agent = NeuralBandit(n_actions, n_features, learning_rate, momentum)

    n_steps = int(Total_steps / Budget)

    cumulative_rewards = np.zeros(Total_steps)
    cumulative_regrets = np.zeros(Total_steps)

    agent = nueral_agent

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
    min_func_val = np.inf

    learning_rates = [0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25]
    momentums = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    comb = len(learning_rates) * len(momentums)

    random.shuffle(learning_rates)
    random.shuffle(momentums)

    selected_alpha = np.zeros((comb,4))
    selected_alpha = pd.DataFrame(selected_alpha)

    for z in range(0, Budget):
        if z <= comb-1:

            index = z
            index1 = np.mod(z, int(len(learning_rates)))
            index2 = int(np.floor(z/ int(len(learning_rates))))

            learning_rate = learning_rates[index1]
            momentum = momentums[index2]

            selected_alpha[0].iloc[index1 + (int(len(learning_rates))) * index2] = selected_alpha[0].iloc[index] + 1
            agent.update_alpha(learning_rate, momentum)

            if z >= 1:
                selected_alpha[1].iloc[index-1] = selected_alpha[1].iloc[index-1] + (1 - Reduced_Vector[9].iloc[z - 1])

        if z > comb-1:
            previous_index = index
            selected_alpha[1].iloc[previous_index] = selected_alpha[1].iloc[previous_index] + (1 - Reduced_Vector[9].iloc[z - 1])

            index = np.argmax((selected_alpha[1] / selected_alpha[0]) + np.sqrt(2 * (math.log10(i)) / selected_alpha[0]))

            selected_alpha[0].iloc[index] = selected_alpha[0].iloc[index] + 1

            index1 = np.mod(index, int(len(learning_rates)))
            index2 = int(np.floor(index / int(len(learning_rates))))

            learning_rate = learning_rates[index1]
            momentum = momentums[index2]

            agent.update_alpha(learning_rate, momentum)

        if(z==Non_Stationary_Step and Non_Stationary == True):
            bandit.theta = New_theta
            bandit.update_theta(bandit.theta)

        for o in range(0, n_steps):

            # if t >= 2*(n_steps/Budget):
            #     if t % (n_steps/Budget) == 0:
            #         alpha = HPO()
            #         linucb_agent.update_alpha(alpha)
            # else:
            #     alpha = initial_alpha

            t = o + z * n_steps

            x = np.random.randn(n_features)
            pred_rewards = agent.predict([x]).numpy()
            Predicted_Rewards.iloc[t] = pred_rewards


            action = np.argmax(pred_rewards)
            # Selected_Actions[action] = Selected_Actions[action] + 1
            reward, optimal_reward, worst_reward = bandit.get_reward(action, x)
            # optimal_reward = bandit.get_optimal_reward(x)
            agent.update(action, x, reward)

            # To Calculate the MAPE of Prediction in each step

            Vector[6].iloc[t] = np.abs((reward - np.argmax(pred_rewards)) / reward)
            Vector[7].iloc[t] = learning_rate
            Vector[5].iloc[t] = momentum

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
        Reduced_Vector[4].iloc[z] = 0
        Reduced_Vector[5].iloc[z] = momentum
        Reduced_Vector[6].iloc[z] = Vector[6].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[8].iloc[z] = Vector[11].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[9].iloc[z] = Vector[12].iloc[int(z*length):int((z+1)*length)].mean()
        Reduced_Vector[10].iloc[z] = z
        Reduced_Vector[11].iloc[z] = learning_rate
        Reduced_Vector[12].iloc[z] = Vector[7].iloc[int(z*length):int((z+1)*length)].mean()
    Reduced_Vector[7] = 0
    Reduced_Vector[7] = Reduced_Vector[12]

    return Reduced_Vector, Vector

bandit = Bandit(n_actions, n_features)
bandit.theta = Main_theta
Reduced_Vector_BOBBOBUCB, Vector_BOBBOBUCB = BOBUCB_Task_Optimization(n_actions, n_features, bandit, initial_alpha, Total_steps, Budget, final_df, Non_Stationary_Step, Non_Stationary, New_theta)


Reduced_Vector_Algorithm.to_excel("Reduced_Vector_Algorithm_Shuffle_Asli_NN_10_10_5.xlsx")
Reduced_Vector_Bayesian.to_excel("Reduced_Vector_Bayesian_Shuffle_Asli_NN_10_10_5.xlsx")
Reduced_Vector_BOBSoftmax.to_excel("Reduced_Vector_BOBSoftmax_Shuffle_Asli_NN_10_10_5.xlsx")
Reduced_Vector_GradualDecresement.to_excel("Reduced_Vector_GradualDecresement_Shuffle_Asli_NN_10_10_5.xlsx")
Reduced_Vector_RandomSearch.to_excel("Reduced_Vector_RandomSearch_Shuffle_Asli_NN_10_10_5.xlsx")
Reduced_Vector_BOBBOBUCB.to_excel("Reduced_Vector_BOBBOBUCB_Shuffle_Asli_NN_10_10_5.xlsx")
