import numpy as np

class MemoryBuffer:
    """Store Experiences For Agent To Sample and Learn From"""

    def __init__(self, max_size, input_shape, n_actions, opt):

        self.mem_size = max_size
        self.mem_counter = 0
        # self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.state_memory = np.zeros(
            (self.mem_size, * (np.prod(input_shape),)))
        self.next_state_memory = np.zeros(
            (self.mem_size, * (np.prod(input_shape),)))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.feature_extractor = get_model(opt)


    def push(self, state, action, reward, next_state, done):
        """Push New Experiences Into Buffer And Increment Counter"""
        index = self.mem_counter % self.mem_size
        state = self.feature_extractor(state)[0]
        next_state = self.feature_extractor(next_state)[0]

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1


    def sample_buffer(self, batch_size):
        """Randomly Sample from Memory Buffer According To Batch Size"""
        max_mem = min(self.mem_counter, self.mem_size)

        # 'False' to prevent repetition in sampling
        batch = np.random.choice(max_mem, batch_size, replace=False)

        state_batch = self.state_memory[batch]
        next_state_batch = self.next_state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        done_batch = self.terminal_memory[batch]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch