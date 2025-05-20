# How to Add a New Environment

1. Create a new directory (e.g., `my_env/`) under the `env/` folder.
2. Implement your environment class in `env.py`. The class should provide:
   - **End-to-end interface:**
     ```python
     # 1. Create the environment
     env = Env(max_turns, n_envs, *args, **kwargs)
     self.trajs = {}  # For logging trajectories

     # 2. Agent interacts with the environment
     filtered_responses_str, feedbacks, dones, step_rewards, env_trajs = env.run(responses_str, batch, chat_template)
     # filtered_responses_str: responses for the next turn
     # chat_template: feedback format (default None)
     # batch: contains info for the environment, e.g., batch.non_tensor_batch['extra_info'] (add info here in `create_dataset.py`)

     # 3. (Optional) Custom reward allocation
     reward_tensors = env.get_reward_allocation(reward_tensors)
     ```

   - **Example of internal structure (optional):**
     ```python
     env.max_turns
     env.reward      # Provide a step reward at each turn. If done, append 0.0.
     env.extract_action()
     env.execute()
     env.finished()  # or self.success() as a stopping criterion
     ```

3. Import your environment class in `env/__init__.py` and add a branch in the `get_env` function.
4. Use `+env.xx` in your training shell script to specify custom arguments for the new environment.