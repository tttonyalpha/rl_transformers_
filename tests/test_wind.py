from environments.Wind.wind_env import WindEnv

env = WindEnv()
obs = env.reset()

for i in range(100):
    action = env.action_space.sample()
    ob, reward, done, info = env.step(action)

    print(action)
    print(type(ob), ob, reward, done)
