import gym
import numpy as np
import matplotlib.pyplot as plt
import copy

NUM_EPISODES = 4000
LEARNING_RATE = 0.000025
GAMMA = 0.99


# noinspection PyMethodMayBeStatic
class Agent:
    def __init__(self):
        self.w = np.random.rand(4, 2)

    def policy(self, state):
        z = state.dot(self.w)
        exp = np.exp(z)
        return exp/np.sum(exp)

    def __softmax_grad(self, softmax):
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)

    def grad(self, probs, action, state):
        dsoftmax = self.__softmax_grad(probs)[action,:]
        dlog = dsoftmax / probs[0,action]
        grad = state.T.dot(dlog[None,:])
        return grad

    def update_with(self, grads, rewards):

        for i in range(len(grads)):
            # Loop through everything that happened in the episode
            # and update towards the log policy gradient times **FUTURE** reward
            self.w += LEARNING_RATE * grads[i] * sum(
                [ r * (GAMMA ** r) for t,r in enumerate(rewards[i:])])



def main(argv):
    env = gym.make('CartPole-v0')
    np.random.seed(1)

    agent = Agent()
    complete_scores = []

    for e in range(NUM_EPISODES):
        state = env.reset()[None, :]

        rewards = []
        grads = []
        score = 0

        while True:

            probs = agent.policy(state)
            action_space = env.action_space.n
            action = np.random.choice(action_space, p=probs[0])

            next_state, reward, done,_ = env.step(action)
            next_state = next_state[None,:]
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
    plt.show()


if __name__ == '__main__':
    main(None)


<<<<<<< HEAD
        # Loop through everything that happend in the episode
        # and update towards the log policy gradient times **FUTURE** reward

        total_grad_effect = 0
        for t, r in enumerate(rewards[i:]):
            total_grad_effect += r * (GAMMA ** r)
        w += LEARNING_RATE * grads[i] * total_grad_effect
=======
>>>>>>> a1799297f2c1c5bc506a15eca1c283252b971b4b


