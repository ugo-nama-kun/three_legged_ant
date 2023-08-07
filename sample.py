import numpy as np
import gymnasium
import tla_env

env = gymnasium.make("NestingTLA-v0", render_mode="human")

print(env.action_space)
print(env.observation_space)

while True:
    env.reset()
    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        
        obs, reward, done, truncated, info = env.step(action)
        done = done | truncated
        
        env.render()
        
        print(steps, ":", reward, info)
    
    print(f"Finish at {steps}")
