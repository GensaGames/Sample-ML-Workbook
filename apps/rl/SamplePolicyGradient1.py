import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import copy

# Hyperparameters
NUM_EPISODES = 4000
LEARNING_RATE = 0.000025
GAMMA = 0.99

# Create gym and seed numpy
env = gym.make('CartPole-v0')
poly = PolynomialFeatures(1)
nA = env.action_space.n
# np.random.seed(1)

# Init weight
w = np.random.rand(5, 1) - 1

# Keep stats for final print of graph
episode_rewards = []


# Our policy that maps state to action parameterized by w
# noinspection PyShadowingNames
def policy(state, w):
    z = np.sum(state.dot(w))
    return sigmoid(z)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_grad(x):
    f = sigmoid(x)
    return f * (1 - f)


# Main loop
# Make sure you update your weights AFTER each episode
for e in range(NUM_EPISODES):

    state = env.reset()[None, :]
    state = poly.fit_transform(state)

    grads = []
    rewards = []

    # Keep track of game score to print
    score = 0

    while True:

        # Uncomment to see your model train in real time (slower)
        # env.render()

        # Sample from policy and take action in environment
        probs = policy(state, w)
        print("Action: " + str(probs))
        action = int(round(probs))
        next_state, reward, done, _ = env.step(action)
        next_state = poly.fit_transform(next_state.reshape(1, 4))

        # Compute gradient and save with reward in memory for our weight updates
        dsoftmax = sigmoid_grad(probs)
        dlog = dsoftmax / probs
        grad = state.T.dot(dlog)
        grad = grad.reshape(5, 1)

        grads.append(grad)
        rewards.append(reward)

        score += reward

        # Dont forget to update your old state to the new state
        state = next_state

        if done:
            break

    # Weight update
    for i in range(len(grads)):

        # Loop through everything that happend in the episode
        # and update towards the log policy gradient times **FUTURE** reward

        total_grad_effect = 0
        for t, r in enumerate(rewards[i:]):
            total_grad_effect += r * (GAMMA ** r)
        w += LEARNING_RATE * grads[i] * total_grad_effect

    # Append for logging and print
    episode_rewards.append(score)
    print("EP: " + str(e) + " Score: " + str(score) + "         ", end="\r", flush=False)

plt.plot(np.arange(NUM_EPISODES),
         episode_rewards)
plt.show()
env.close()
