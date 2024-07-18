from environments.Numpad import numpad_discrete# Numpad2DDiscrete

env = numpad_discrete.Environment(numpad_discrete.Config)

# print(env.__dict__)

print(env.action_space)

for i in range(10):
    action = env.action_space.sample()
    rt = env.step(action)

    print(action, rt)