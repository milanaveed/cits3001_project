import pandas as pd
import matplotlib.pyplot as plt


# #############################################################################
# # Plotting Reward over Time Steps
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
# # Plotting Distance over Time Steps
# #############################################################################
# # Read data
# x_pos_log = pd.read_csv("./reinforcement-learning/train1-1-v302/x_position_log.csv", index_col='timesteps')
# x_pos_log.plot()
# # Customize the plot
# plt.title('X Position over Time Steps')
# plt.xlabel('Number of Time Steps')
# plt.ylabel('Value')
# # plt.grid(True)
# plt.tight_layout()

# # Display the plot
# plt.show()


#############################################################################
# Plotting Training Time over Time Steps
#############################################################################
# Read data
training_time_log = pd.read_csv("./reinforcement-learning/train1-2-v3/training_time_log.csv", index_col='timesteps')
training_time_log.plot()
# Customize the plot
plt.title('Training Time over Time Steps for 1-2-v3')
plt.xlabel('Number of Time Steps')
plt.ylabel('Minutes')
plt.tight_layout()

# Display the plot
plt.show()


#############################################################################
# Plotting Training Time over Time Steps
#############################################################################
# Read data
pass_rate_log = pd.read_csv("./reinforcement-learning/train1-2-v3/pass_rate_log.csv", index_col='timesteps')
pass_rate_log.plot()
# Customize the plot
plt.title('Pass Rate over Time Steps for 1-2-v3')
plt.xlabel('Number of Time Steps')
plt.ylabel('Pass Rate (%)')
plt.tight_layout()

# Display the plot
plt.show()