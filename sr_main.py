import json
import mazes
import sys, os
import argparse
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from collections import OrderedDict

MAZE_R1 = ['#####',
 		   '#P  #',
 		   '#   #',
		   '# G #',
		   '#####']

MAZE_R2 = ['#####',
 		   '#P  #',
 		   '#  G#',
		   '#   #',
		   '#####']

SIMPLE_MAZE = ['####',
	 		   '#P #',
			   '# G#',
			   '####']

def flat2xy(f):
	f = f + 1 # f indexes from 0, gridworld indexes from 1
	y = f%ncol
	x = (f-y)/ncol + 1
	return x, y

def xy2flat(x,y):
	f = (x-1)*9+y
	f = f - 1 # f indexes from 0, gridworld indexes from 1
	return f

# Take a random action
def random_policy(rnd):
	action = rnd.randint(0,4,1)[0]
	return action

# e-greedy algorithm for choosing next action
def eps_greedy(S, Q, eps, rnd1, rnd2, rnd4):

	uni_rnd = rnd1.uniform(0.0,1.0)
	if uni_rnd > eps:
		# check for highest Q
		# Break ties with coin flip
		action = rnd4.choice(np.flatnonzero(Q[S,:] == Q[S,:].max()))
		# action = Q[S,:].argmax()
	else:
		# choose from our 4 different possible actions randomly:
		action = rnd2.randint(0,4,1)[0]

	return action

def parse_obs(obs, nrow, ncol):

	state_mtx = np.array(obs.layers['P'], dtype=np.float)
	state_mtx = state_mtx[1:nrow+1,1:ncol+1].flatten()

	return state_mtx.argmax()


def setup_maze(maze_type, start_row, start_col):

	if maze_type == 'SIMPLE_MAZE':
		maze_init = mazes.make_maze(SIMPLE_MAZE)
		maze_update = mazes.make_maze(SIMPLE_MAZE)
	elif maze_type == 'MAZE_R1':
		maze_init = mazes.make_maze(MAZE_R1)
		maze_update = mazes.make_maze(MAZE_R2)

	# Place engines in play mode
	maze_init.its_showtime()
	maze_update.its_showtime()

	# Move agent to starting position
	maze_init._sprites_and_drapes['P']._teleport((start_row, start_col))
	maze_update._sprites_and_drapes['P']._teleport((start_row, start_col))

	return maze_init, maze_update

# Indicator Function
def indicator(s, j):
	if s == j:
		return 1
	else:
		return 0

# Track2 Q1
def run_learning_sr(config):

	# Initializations
	maze_type = config['maze_type']
	terminal_step = config['terminal_step']

	episode_len = config['episode_length']
	nrow = config['maze_params']['row']
	ncol = config['maze_params']['col']
	start_row = config['maze_params']['start_row']
	start_col = config['maze_params']['start_col']

	alpha = config['learning_alg_params']['alpha']
	gamma = config['learning_alg_params']['gamma']

	state_len = nrow*ncol
	action_len = 4
	reward_len = 2

	# Don't need all these guys
	rnd = np.random.RandomState(24)

	S = xy2flat(start_row, start_col)
	S_prime = S

	# Initialize mazes (r_1, r_2) and agent at starting position
	maze_init, _ = setup_maze(maze_type, start_row, start_col)
	curr_maze = maze_init

	step = 0
	episode = 0
	episode_step = 0
	result = OrderedDict()

	cum_reward = 0
	cum_reward_lst = []

	Phi_pi = np.zeros((state_len, state_len))
	V_pi = np.zeros((state_len,1))
	result['config'] = config


	while step < terminal_step:

		# Reset episode:
		if episode_step >= episode_len:
			S = xy2flat(start_row, start_col)
			S_prime = S
			maze_init, _ = setup_maze(maze_type, start_row, start_col)
			curr_maze = maze_init # only doing this to reset the agent to the starting position, the next if statement will actually correct the map if need be

			episode_step = 0
			episode += 1

		if curr_maze._game_over:
			A = random_policy(rnd)
			R = 0.
			S_prime = S
		else:
			# Select Action A using a uniform random policy
			A = random_policy(rnd)

			# Apply action A to current maze, get reward R, and new state S'
			obs, R, _ = curr_maze.play(A)
			S_prime = parse_obs(obs, nrow, ncol)

		for j in range(state_len):
			Phi_pi[S, j] = Phi_pi[S, j] + alpha*(indicator(S,j) + gamma*Phi_pi[S_prime, j] - Phi_pi[S, j])
			V_pi[S] = V_pi[S] + alpha*(R + gamma*V_pi[S_prime] - V_pi[S])

		experience = {
			'S' : S,
			'A' : A,
			'R' : R,
			'S_prime' : S_prime
		}
		result[step] = {
			'Phi_pi' : Phi_pi.copy(),
			'V_pi' : V_pi.copy(),
			'experience': experience

		}

		# Update to new state
		S = S_prime

		step += 1
		episode_step += 1

	return result

# Switching for later
def run_experiment(config):

	# Initializations
	terminal_step = config['terminal_step']
	switch_step = config['switch_reward_at_step']

	nrow = config['maze_params']['row']
	ncol = config['maze_params']['col']
	start_row = config['maze_params']['start_row']
	start_col = config['maze_params']['start_col']

	eps = config['policy_params']['epsilon']
	alpha = config['learning_alg_params']['alpha']
	gamma = config['learning_alg_params']['gamma']

	state_len = nrow*ncol
	action_len = 4
	reward_len = 2

	# Don't need all these guys
	rnd1 = np.random.RandomState(24)


	S = xy2flat(start_row, start_col)
	S_prime = S

	# Initialize mazes (r_1, r_2) and agent at starting position
	maze_init, maze_update = setup_maze(start_row, start_col)
	curr_maze = maze_init

	step = 0
	episode = 0
	result = OrderedDict()

	cum_reward = 0
	cum_reward_lst = []

	while step < terminal_step:

		# Reset episode:
		if curr_maze._game_over:
			S = xy2flat(start_row, start_col)
			S_prime = S
			maze_init, maze_update = setup_maze(start_row, start_col)
			curr_maze = maze_init # only doing this to reset the agent to the starting position, the next if statement will actually correct the map if need be

			episode += 1

		# The maze evolves after switch_step:
		if step <= switch_step:
			curr_maze = maze_init
		else:
			old_row, old_col = curr_maze._sprites_and_drapes['P']._virtual_row, \
							   curr_maze._sprites_and_drapes['P']._virtual_col

			maze_update._sprites_and_drapes['P']._teleport((old_row, old_col))
			curr_maze = maze_update

		# Get current state S
		S = S_prime

		# Select Action A using epsilon-greedy (given S, Q)
		A = eps_greedy(S, Q, eps, rnd1, rnd2, rnd4)

		# Apply action A to current maze, get reward R, and new state S'
		obs, R, _ = curr_maze.play(A)
		S_prime = parse_obs(obs)

		# Update Q function
		Q[S,A] = Q[S,A] + alpha*( R + gamma*Q[S_prime,:].max() - Q[S,A] )

		# Update Model with R, S_prime for a particular state action pair
		model[(S,A)] = (R, S_prime)

		if arch == 'dyna_q_plus':
			visited_step[(S,A)] = step

		# Loop sim_epoch times (simulation):
		for i in range(sim_epoch):
			# Get random previously observed state S
			# Get random previously taken action A for that state S
			rnd_S, rnd_A = model.keys()[ rnd3.randint(0, len(model), 1)[0] ]

			# Extract reward R and next state S' for that action A from Model
			sim_R, sim_S_prime = model[(rnd_S, rnd_A)]

			if arch == 'dyna_q_plus':
				tau = step - visited_step[(rnd_S, rnd_A)]
				sim_R += kappa*sqrt(tau)

			# Update Q function
			Q[rnd_S, rnd_A] = Q[rnd_S,rnd_A] + alpha*( sim_R + gamma*Q[sim_S_prime,:].max() - Q[rnd_S, rnd_A] )


		cum_reward += R
		cum_reward_lst.append(cum_reward)

		experience = {
			'S' : S,
			'A' : A,
			'R' : R,
			'S_prime' : S_prime
		}
		result[step] = {
			'config' : config,
			'episode' : episode,
			'experience' : experience,
			'value_function' : Q.copy(),
			'observation' : obs
		}

		step += 1

	return result


def main(argv):

	parser = argparse.ArgumentParser()
	parser.add_argument('config_file', type=str,  help="Configuration file path. e.g. blocking.config")
	args = parser.parse_args()

	#Reading config file
	config_file_path = args.config_file

	if not os.path.exists(config_file_path): raise argparse.ArgumentTypeError('Not a valid config file')

	with open(config_file_path, 'r') as config_fd:
		config = json.load(config_fd)

	result = run_learning_sr(config)
	# cum_reward = 0
	# cum_reward_lst = []
	# for i in range(len(result)):
	# 	cum_reward += result[i]['experience']['R']
	# 	cum_reward_lst.append(cum_reward)
	#
	# print("Cumulative reward: " + str(cum_reward_lst[-1]))
	# plt.plot(cum_reward_lst)
	# plt.ylabel('Cumulative Rewards')
	# plt.xlabel('Step number')
	# plt.show()

if __name__ == '__main__':
	main(sys.argv)
