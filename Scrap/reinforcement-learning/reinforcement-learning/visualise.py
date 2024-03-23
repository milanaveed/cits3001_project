import pandas as pd
import matplotlib.pyplot as plt


# #############################################################################
# # Plotting Reward over Time Steps 1-1 v3
# #############################################################################
# # Read data
# reward_log = pd.read_csv("./reinforcement-learning/train1-1-v302/reward_log.csv", index_col='timesteps')
# reward_log.plot()
# # Customize the plot
# plt.title('Reward over Time Steps')
# plt.xlabel('Number of Time Steps')
# plt.ylabel('Value')
# # plt.grid(True)
# plt.tight_layout()

# # Display the plot
# plt.show()


# #############################################################################
# # Plotting Distance over Time Steps 1-1 v3
# #############################################################################
# # Read data
# x_pos_log = pd.read_csv("./reinforcement-learning/train1-1-v302/x_position_log.csv", index_col='timesteps')
# print('best epoch:', x_pos_log['average_x_position'].idxmax())
# x_pos_log.plot()
# # Customize the plot
# plt.title('X Position over Time Steps')
# plt.xlabel('Number of Time Steps')
# plt.ylabel('Value')
# # plt.grid(True)
# plt.tight_layout()

# # Display the plot
# plt.show()


# #############################################################################
# # Plotting Training Time over Time Steps for 1-1 v3 improved and early version
# #############################################################################
# # Read data
# training_time_log1 = pd.read_csv("./reinforcement-learning/train1-1-99/training_time_log.csv", index_col='timesteps')
# training_time_log2 = pd.read_csv("./reinforcement-learning/train1-1-v303/training_time_log.csv", index_col='timesteps')

# training_time_log = pd.merge(training_time_log1, training_time_log2, on='timesteps', how='inner')

# training_time_log.plot()
# # Customize the plot
# plt.title('Training Time over Time Steps for 1-1-v3')
# plt.xlabel('Number of time steps')
# plt.ylabel('Minutes')
# plt.tight_layout()

# # Display the plot
# plt.show()


#############################################################################
# Plotting Training Time over Time Steps for 1-1 v3 improved - assumed
#############################################################################
# Read data
# training_time_log2 = pd.read_csv("./reinforcement-learning/train1-1-v303/training_time_log.csv", index_col='timesteps')


# training_time_log2.plot()
# # Customize the plot
# plt.title('Training Time over Time Steps for 1-1-v3')
# plt.xlabel('Number of time steps')
# plt.ylabel('Minutes')
# plt.tight_layout()

# # Display the plot
# plt.show()




# #############################################################################
# # Plotting Training Time over Time Steps 1-2-v3
# #############################################################################
# Read data
# training_time_log = pd.read_csv("./reinforcement-learning/train1-2-v3/training_time_log.csv", index_col='timesteps')
# training_time_log.plot()
# # Customize the plot
# plt.title('Training Time over Time Steps for 1-2-v3')
# plt.xlabel('Number of Time Steps')
# plt.ylabel('Minutes')
# plt.tight_layout()

# # Display the plot
# plt.show()


# #############################################################################
# # Plotting Distance over Time Steps 1-2 v3
# #############################################################################
# # Read data
# x_pos_log = pd.read_csv("./reinforcement-learning/train1-2-v3/x_position_log.csv", index_col='timesteps')
# print('best epoch:', x_pos_log['average_distance'].idxmax())
# x_pos_log.plot()
# # Customize the plot
# plt.title('X Position over Time Steps for 1-2-v3')
# plt.xlabel('Number of Time Steps')
# plt.ylabel('Value')
# # plt.grid(True)
# plt.tight_layout()

# # Display the plot
# plt.show()


# #############################################################################
# # Plotting pass rate 1-2-v3
# #############################################################################
# Read data
# pass_rate_log = pd.read_csv("./reinforcement-learning/train1-2-v3/pass_rate_log.csv", index_col='timesteps')
# print('best epoch:', pass_rate_log['pass_rate(%)'].idxmax())

# pass_rate_log.plot()
# # Customize the plot
# plt.title('Pass Rate over Time Steps for 1-2-v3')
# plt.xlabel('Number of Time Steps')
# plt.ylabel('Pass Rate (%)')
# plt.tight_layout()

# # # Display the plot
# plt.show()


# #############################################################################
# # Plotting Training Time over Time Steps 1-1-v0
# #############################################################################
# Read data
# training_time_log = pd.read_csv("./reinforcement-learning/train1-1-v0/training_time_log.csv", index_col='timesteps')
# training_time_log.plot()
# # Customize the plot
# plt.title('Training Time over Time Steps for 1-1-v0')
# plt.xlabel('Number of Time Steps')
# plt.ylabel('Minutes')
# plt.tight_layout()

# # Display the plot
# plt.show()


# #############################################################################
# # Plotting Distance over Time Steps 1-1 v0
# #############################################################################
# Read data
# x_pos_log = pd.read_csv("./reinforcement-learning/train1-1-v0/x_position_log.csv", index_col='timesteps')
# print('best epoch:', x_pos_log['average_distance'].idxmax())
# x_pos_log.plot()
# # Customize the plot
# plt.title('X Position over Time Steps for 1-1-v0')
# plt.xlabel('Number of Time Steps')
# plt.ylabel('Value')
# # plt.grid(True)
# plt.tight_layout()

# # Display the plot
# plt.show()


# #############################################################################
# # Plotting pass rate 1-1-v0
# #############################################################################
# Read data
# pass_rate_log = pd.read_csv("./reinforcement-learning/train1-1-v0/pass_rate_log.csv", index_col='timesteps')
# print('best epoch:', pass_rate_log['pass_rate(%)'].idxmax())

# pass_rate_log.plot()
# # Customize the plot
# plt.title('Pass Rate over Time Steps for 1-1-v0')
# plt.xlabel('Number of Time Steps')
# plt.ylabel('Pass Rate (%)')
# plt.tight_layout()

# # # Display the plot
# plt.show()



# #############################################################################
# # Plotting Distance over Time Steps transfer learning
# #############################################################################
# Read data
# x_pos_log = pd.read_csv("./reinforcement-learning/transfer-learning-1-1-v0/x_position_log.csv", index_col='timesteps')
# x_pos_log.plot()
# # Customize the plot
# plt.title('X Position over Time Steps')
# plt.xlabel('Number of Time Steps')
# plt.ylabel('Value')
# # plt.grid(True)
# plt.tight_layout()

# # Display the plot
# plt.show()

# pass_rate_log = pd.read_csv("./reinforcement-learning/transfer-learning-1-1-v0/pass_rate_log.csv", index_col='timesteps')
# pass_rate_log.plot()
# # Customize the plot
# plt.title('Pass Rate over Time Steps')
# plt.xlabel('Number of Time Steps')
# plt.ylabel('Percentage')
# # plt.grid(True)
# plt.tight_layout()

# # Display the plot
# plt.show()
