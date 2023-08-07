from gymnasium.envs.registration import register

register(
    id='TLA-v0',
    entry_point='tla_env.envs:ThreeLeggedAntEnv',
)

register(
    id='NestingTLA-v0',
    entry_point='tla_env.envs:NestingThreeLeggedAntEnv',
)

register(
    id='CommandTLA-v0',
    entry_point='tla_env.envs:CommandThreeLeggedAntEnv',
)
