import morality_gym as mg

make_kwargs = {}  #
env, mc = mg.make(env_id="...", mc_id="...", **make_kwargs)
def random_policy(obs):
    return env.action_space.sample()

# Evaluate morality_metric for morality chain 'mc', (wrapped) environment 'env' and policy 'random_policy'
mm, info = mc.morality_metric(env, random_policy, ...)