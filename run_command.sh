
python ppo_continuous_action.py \
--env-id CommandTLA-v0 \
--max-steps 150 \
--max-test-steps 150 \
--track \
--wandb-group Command \
--render-test
