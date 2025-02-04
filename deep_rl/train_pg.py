"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Michael Chang and Soroush Nasiriany
Adapted for CS 182/282A Spring 2019 by Daniel Seita
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gym
import logz
import os
import time
import inspect
import sys
from multiprocessing import Process


def build_mlp(input_placeholder, output_size, scope, n_layers, size,
              activation=tf.tanh, output_activation=None):
    """Builds a feedforward neural network

    arguments:
        input_placeholder: placeholder variable for the state (batch_size, input_size)
        output_size: size of the output layer
        scope: variable scope of the network
        n_layers: number of hidden layers (does not count output layer)
        size: dimension of the hidden layer
        activation: activation of the hidden layers
        output_activation: activation of the ouput layers

    returns:
        output placeholder of the network (the result of a forward pass)
    """
    with tf.variable_scope(scope):
        x = input_placeholder
        for _ in range(n_layers):
            x = tf.layers.dense(x, size, activation=activation)
        output_placeholder = tf.layers.dense(x, output_size, activation=output_activation)
    return output_placeholder


def pathlength(path):
    return len(path["reward"])


def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)



class Agent(object):

    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']
        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']
        self.gamma = estimate_return_args['gamma']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.normalize_advantages = estimate_return_args['normalize_advantages']

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    #========================================================================================#
    #                           ----------PROBLEM 1----------
    #========================================================================================#
    def define_placeholders(self):
        """
        Placeholders for batch batch observations / actions / advantages in
        policy gradient loss function.  See Agent.build_computation_graph for
        notation

        returns:
            sy_ob_no: placeholder for observations
            sy_ac_na: placeholder for actions
            sy_adv_n: placeholder for advantages
        """
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)
        # ------------------------------------------------------------------
        # START OF YOUR CODE
        # ------------------------------------------------------------------
        sy_adv_n = tf.placeholder(shape=[None], name='ad', dtype=tf.float32)
        # ------------------------------------------------------------------
        # END OF YOUR CODE
        # ------------------------------------------------------------------
        return sy_ob_no, sy_ac_na, sy_adv_n


    #========================================================================================#
    #                           ----------PROBLEM 1----------
    #========================================================================================#
    def policy_forward_pass(self, sy_ob_no):
        """
        Constructs the symbolic operation for the policy network outputs,
        which are the parameters of the policy distribution p(a|s).

        arguments:
            sy_ob_no: (batch_size, self.ob_dim)

        returns:
            the parameters of the policy.

            if discrete, the parameters are the logits of a categorical distribution
                over the actions
                sy_logits_na: (batch_size, self.ac_dim)

            if continuous, the parameters are a tuple (mean, log_std) of a Gaussian
                distribution over actions. log_std should just be a trainable
                variable, not a network output.
                sy_mean: (batch_size, self.ac_dim)
                sy_logstd: (self.ac_dim,)

        Hint: use the 'build_mlp' function to output the logits (in the discrete case)
            and the mean (in the continuous case).
            Pass in self.n_layers for the 'n_layers' argument, and
            pass in self.size for the 'size' argument.
        """
        if self.discrete:
            # ------------------------------------------------------------------
            # START OF YOUR CODE
            # ------------------------------------------------------------------
            sy_logits_na = build_mlp(input_placeholder=sy_ob_no, output_size=self.ac_dim, scope="pol", n_layers=self.n_layers, size=self.size, activation=tf.tanh, output_activation=None)
            # ------------------------------------------------------------------
            # END OF YOUR CODE
            # ------------------------------------------------------------------
            return sy_logits_na 
        else:
            # ------------------------------------------------------------------
            # START OF YOUR CODE
            # ------------------------------------------------------------------
            sy_mean = build_mlp(input_placeholder=sy_ob_no, output_size=self.ac_dim, scope="pol", n_layers=self.n_layers, size=self.size, activation=tf.tanh, output_activation=None)
            sy_logstd = tf.Variable(tf.zeros(shape=[self.ac_dim]), dtype=tf.float32)
            # ------------------------------------------------------------------
            # END OF YOUR CODE
            # ------------------------------------------------------------------
            return (sy_mean, sy_logstd)

    #========================================================================================#
    #                           ----------PROBLEM 1----------
    #========================================================================================#
    def sample_action(self, policy_parameters):
        """
        Constructs a symbolic operation for stochastically sampling from the
        policy distribution

        arguments:
            policy_parameters
                if discrete: logits of a categorical distribution over actions
                    sy_logits_na: (batch_size, self.ac_dim)
                if continuous: (mean, log_std) of a Gaussian distribution over actions
                    sy_mean: (batch_size, self.ac_dim)
                    sy_logstd: (self.ac_dim,)

        returns:
            sy_sampled_ac:
                if discrete: (batch_size,)
                if continuous: (batch_size, self.ac_dim)

        Hint: for the continuous case, use the reparameterization trick:
             The output from a Gaussian distribution with mean 'mu' and std 'sigma' is

                  mu + sigma * z,         z ~ N(0, I)

             This reduces the problem to just sampling z.
        """
        if self.discrete:
            sy_logits_na = policy_parameters
            # ------------------------------------------------------------------
            # START OF YOUR CODE
            # ------------------------------------------------------------------
            sy_sampled_ac = tf.reshape(tf.random.categorical(logits=sy_logits_na, num_samples=1), [-1])
            # ------------------------------------------------------------------
            # END OF YOUR CODE
            # ------------------------------------------------------------------
        else:
            sy_mean, sy_logstd = policy_parameters
            # ------------------------------------------------------------------
            # START OF YOUR CODE
            # ------------------------------------------------------------------
            sy_sampled_ac = sy_mean + tf.exp(sy_logstd) * tf.random.normal(shape=tf.shape(sy_mean))
            # ------------------------------------------------------------------
            # END OF YOUR CODE
            # ------------------------------------------------------------------
        return sy_sampled_ac

    #========================================================================================#
    #                           ----------PROBLEM 1----------
    #========================================================================================#
    def get_log_prob(self, policy_parameters, sy_ac_na):
        """
        Constructs a symbolic operation for computing the log probability of a
        set of actions that were actually taken according to the policy

        arguments:
            policy_parameters
                if discrete: logits of a categorical distribution over actions
                    sy_logits_na: (batch_size, self.ac_dim)
                if continuous: (mean, log_std) of a Gaussian distribution over actions
                    sy_mean: (batch_size, self.ac_dim)
                    sy_logstd: (self.ac_dim,)

            sy_ac_na:
                if discrete: (batch_size,)
                if continuous: (batch_size, self.ac_dim)

        returns:
            sy_logprob_n: (batch_size)

        Hint:
            For the discrete case, use the log probability under a categorical distribution.
            For the continuous case, use the log probability under a multivariate gaussian.
        """
        if self.discrete:
            sy_logits_na = policy_parameters
            # ------------------------------------------------------------------
            # START OF YOUR CODE
            # ------------------------------------------------------------------
            dist = tfp.distributions.Categorical(logits=sy_logits_na,dtype=tf.int32)
            sy_logprob_n = dist.log_prob(sy_ac_na)
            # ------------------------------------------------------------------
            # END OF YOUR CODE
            # ------------------------------------------------------------------
        else:
            sy_mean, sy_logstd = policy_parameters
            # ------------------------------------------------------------------
            # START OF YOUR CODE
            # ------------------------------------------------------------------
            dist = tfp.distributions.MultivariateNormalDiag(loc=sy_mean, scale_diag=tf.exp(sy_logstd))
            sy_logprob_n = dist.log_prob(sy_ac_na)
            # ------------------------------------------------------------------
            # END OF YOUR CODE
            # ------------------------------------------------------------------
        return sy_logprob_n


    def build_computation_graph(self):
        """
        Symbolic variables have the prefix sy_, to distinguish them from the numerical values
        that are computed later in the function

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
        is None

        ----------------------------------------------------------------------------------
        loss: a function of self.sy_logprob_n and self.sy_adv_n that we will differentiate
            to get the policy gradient.
        """
        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n = self.define_placeholders()

        # The policy takes in an observation and produces a distribution over the action space
        self.policy_parameters = self.policy_forward_pass(self.sy_ob_no)

        # We can sample actions from this action distribution.
        # This will be called in Agent.sample_trajectory() where we generate a rollout.
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        #========================================================================================#
        #                           ----------PROBLEM 1----------
        # Loss Function and Training Operation
        #========================================================================================#
        # ------------------------------------------------------------------
        # START OF YOUR CODE
        # ------------------------------------------------------------------
        losses = self.sy_logprob_n * self.sy_adv_n
        self.loss = -1 * tf.reduce_mean(losses)
        # ------------------------------------------------------------------
        # END OF YOUR CODE
        # ------------------------------------------------------------------
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def sample_trajectories(self, itr, env):
        """Collect paths until we have enough timesteps, as determined by the
        length of all paths collected in this batch.
        """
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch


    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards = [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            #====================================================================================#
            #                           ----------PROBLEM 2----------
            #====================================================================================#
            # ------------------------------------------------------------------
            # START OF YOUR CODE
            # ------------------------------------------------------------------
            feed = {self.sy_ob_no:[ob]}
            ac = self.sess.run(self.sy_sampled_ac, feed_dict=feed)
            # ------------------------------------------------------------------
            # END OF YOUR CODE
            # ------------------------------------------------------------------
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"observation" : np.array(obs, dtype=np.float32),
                "reward" : np.array(rewards, dtype=np.float32),
                "action" : np.array(acs, dtype=np.float32)}
        return path

    #====================================================================================#
    #                           ----------PROBLEM 2----------
    #====================================================================================#
    def sum_of_rewards(self, re_n):
        """ Monte Carlo estimation of the Q function.

        let sum_of_path_lengths be the sum of the lengths of the paths sampled from
            Agent.sample_trajectories
        let num_paths be the number of paths sampled from Agent.sample_trajectories

        arguments:
            re_n: length: num_paths. Each element in re_n is a numpy array
                containing the rewards for the particular path

        returns:
            q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                whose length is the sum of the lengths of the paths
        ----------------------------------------------------------------------------------

        Your code should construct numpy arrays for Q-values which will be used to compute
        advantages (which will in turn be fed to the placeholder you defined in
        Agent.define_placeholders).

        Recall that the expression for the policy gradient PG is

              PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t)]

        where

              tau=(s_0, a_0, ...) is a trajectory,
              Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
              and b_t is a baseline which may depend on s_t.

        You will write code for two cases, controlled by the flag 'reward_to_go':

          Case 1: trajectory-based PG  (reward_to_go = False)

              Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
              entire trajectory (regardless of which time step the Q-value should be for).

              For this case, the policy gradient estimator is

                  E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]

              where

                  Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.

              Thus, you should compute

                  Q_t = Ret(tau)

          Case 2: reward-to-go PG  (reward_to_go = True)

              Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
              from time step t. Thus, you should compute

                  Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}

        Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        like the 'ob_no' and 'ac_na' above.
        """
        # ------------------------------------------------------------------
        # START OF YOUR CODE
        # ------------------------------------------------------------------
        num_paths = len(re_n)
        sum_of_path_lengths = np.sum([r.shape[0] for r in re_n])
        q_n = np.zeros(sum_of_path_lengths)
        
        idx = 0
        if not self.reward_to_go: #each = sum of all
            for re in re_n:
                discounts = np.logspace(self.gamma, self.gamma**len(re),num=len(re), endpoint=True)
                discounted_re = discounts * re
                ret_tau = np.sum(discounted_re)
                q_n[idx:idx+len(re)] = ret_tau
                idx += len(re)
        else:    
            for re in reversed(re_n):
                q_n[idx] = re[-1]
                idx += 1
                for r in reversed(re[:-1]): 
                    q_n[idx] = q_n[idx - 1] * self.gamma + r
                    idx += 1
            q_n = q_n[::-1]
        # ------------------------------------------------------------------
        # END OF YOUR CODE
        # ------------------------------------------------------------------
        return q_n


    def compute_advantage(self, ob_no, q_n):
        """For CS 182/282A, we just use `q_n` for the advantages.

        In CS 294-112, you'll learn in more detail about how to reduce variance
        in policy gradient updates. For simplicity we won't implement this here.
        """
        adv_n = q_n.copy()
        return adv_n


    def estimate_return(self, ob_no, re_n):
        """ Estimates the returns over a set of trajectories.

        let sum_of_path_lengths be the sum of the lengths of the paths sampled from
            Agent.sample_trajectories
        let num_paths be the number of paths sampled from Agent.sample_trajectories

        arguments:
            ob_no: shape: (sum_of_path_lengths, ob_dim)
            re_n: length: num_paths. Each element in re_n is a numpy array
                containing the rewards for the particular path

        returns:
            q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                whose length is the sum of the lengths of the paths
            adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                advantages whose length is the sum of the lengths of the paths
        """
        q_n = self.sum_of_rewards(re_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        #====================================================================================#
        #                           ----------PROBLEM 2----------
        # Advantage Normalization
        #====================================================================================#
        if self.normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            # ------------------------------------------------------------------
            # START OF YOUR CODE
            # ------------------------------------------------------------------
            mean, std = 0, 1
            z = (adv_n - np.mean(adv_n)) / (np.std(adv_n))
            adv_n = mean + std * z 
            # ------------------------------------------------------------------
            # END OF YOUR CODE
            # ------------------------------------------------------------------
        return q_n, adv_n


    def update_parameters(self, ob_no, ac_na, q_n, adv_n):
        """
        Update the parameters of the policy and (possibly) the neural network baseline,
        which is trained to approximate the value function.

        arguments:
            ob_no: shape: (sum_of_path_lengths, ob_dim)
            ac_na: shape: (sum_of_path_lengths).
            q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                whose length is the sum of the lengths of the paths
            adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                advantages whose length is the sum of the lengths of the paths

        returns:
            nothing
        """
        #====================================================================================#
        #                           ----------PROBLEM 2----------
        #====================================================================================#
        # Performing the Policy Update
        #
        # Call the update operation necessary to perform the policy gradient update based on
        # the current batch of rollouts.
        #
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below.
        # ------------------------------------------------------------------
        # START OF YOUR CODE
        # ------------------------------------------------------------------
        feed = {self.sy_ob_no : ob_no, self.sy_ac_na : ac_na, self.sy_adv_n : adv_n}
        self.sess.run(self.update_op, feed_dict=feed)
        # ------------------------------------------------------------------
        # END OF YOUR CODE
        # ------------------------------------------------------------------


def train_PG(
        exp_name,
        env_name,
        n_iter,
        gamma,
        min_timesteps_per_batch,
        max_path_length,
        learning_rate,
        reward_to_go,
        animate,
        logdir,
        normalize_advantages,
        seed,
        n_layers,
        size):
    start = time.time()
    setup_logger(logdir, locals())

    #========================================================================================#
    # Set Up Env
    #========================================================================================#

    # Make the gym environment
    env = gym.make(env_name)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    # Initialize Agent
    #========================================================================================#
    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
    }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma': gamma,
        'reward_to_go': reward_to_go,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)

    # build computation graph
    agent.build_computation_graph()

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by
        # concatenating across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = [path["reward"] for path in paths]

        q_n, adv_n = agent.estimate_return(ob_no, re_n)
        agent.update_parameters(ob_no, ac_na, q_n, adv_n)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    args = parser.parse_args()

    if not(os.path.exists('data_pg')):
        os.makedirs('data_pg')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data_pg', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None
    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                seed=seed,
                n_layers=args.n_layers,
                size=args.size
            )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
