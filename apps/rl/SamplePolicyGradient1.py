import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import copy

NUM_EPISODES = 4000
LEARNING_RATE = 0.000025
GAMMA = 0.99


# noinspection PyMethodMayBeStatic
class Agent:
    def __init__(self):
        self.poly = PolynomialFeatures(1)
        self.w = np.random.rand(4, 1) - 1

    # Our policy that maps state to action parameterized by w
    # noinspection PyShadowingNames
    def policy(self, state):
        z = np.sum(state.dot(self.w))
        return self.sigmoid(z)

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def sigmoid_grad(self, x):
        f = self.sigmoid(x)
        return f * (1 - f)

    def grad(self, probs, action, state):
        dsoftmax = self.sigmoid_grad(probs)
        dlog = dsoftmax / probs
        grad = state.T.dot(dlog)
        grad = grad.reshape(4, 1)
        return grad

    def update_with(self, grads, rewards):

        for i in range(len(grads)):
            # Loop through everything that happend in the episode
            # and update towards the log policy gradient times **FUTURE** reward

            total_grad_effect = 0
            for t, r in enumerate(rewards[i:]):
                total_grad_effect += r * (GAMMA ** r)
            self.w += LEARNING_RATE * grads[i] * total_grad_effect


def main(argv):
    env = gym.make('CartPole-v0')

    agent = Agent()
    complete_scores = []

    for e in range(NUM_EPISODES):
        state = env.reset()[None, :]
        # state = agent.poly.fit_transform(state)

        rewards = []
        grads = []
        score = 0

        while True:

            probs = agent.policy(state)
            action = int(round(probs))

            next_state, reward, done, _ = env.step(action)
            next_state = next_state[None, :]
            # next_state = agent.poly.fit_transform(next_state.reshape(1, 4))

            grad = agent.grad(probs, action, state)
            grads.append(grad)
            rewards.append(reward)

            score += reward
            state = next_state

            if done:
                break

        agent.update_with(grads, rewards)
        complete_scores.append(score)

    env.close()
    plt.plot(np.arange(NUM_EPISODES),
             complete_scores)
    plt.savefig('image1.png')


if __name__ == '__main__':
    main(None)


