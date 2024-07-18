from environments.ballet_env_old import BalletWrapper

env = BalletWrapper()


vis_obs, vec_obs = env.reset()


# print(env.__dict__)
print(env._vector_observatoin_space)
print(env._visual_observation_space)
print(env.action_space.sample())
# print(vis_obs, vis_obs.shape)
# print(vec_obs, vec_obs.shape)


# print(env.obsevation_space())
# print(env.action_space())

for i in range(300):
    action = env.action_space.sample()
    vis_obs, vec_obs, reward, done, info = env.step([action])
    print(action, vec_obs, reward, done, info)
