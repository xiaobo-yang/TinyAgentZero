# export before `ray start`
export PYTHONHASHSEED=10000
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS

N_GPUS_PER_NODE=8
N_NODES=1
LOGGER=wandb

ENV=search
DATA_PATH=dataset/$ENV
MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B
MODEL_PATH=/home/yangxiaobo/my_data/models/$MODEL_NAME
PROJ_NAME=agent-zero
RUN_TIME=$(date +%Y%m%d_%H%M%S)
EXP_NAME=$ENV-$MODEL_NAME-PPO-${RUN_TIME}-rephrase-sft-multi-val
git add .
git commit -m "run $EXP_NAME"
GIT_HASH=$(git rev-parse --short=8 HEAD)
EXP_NAME=${EXP_NAME}-${GIT_HASH}
LOG_FILE_PATH=log/$EXP_NAME
mkdir -p $LOG_FILE_PATH

N_AGENTS=1
MAX_TURNS=4
MAX_VAL_TURNS=4
TURNS_MAXIMUM=$(( MAX_TURNS > MAX_VAL_TURNS ? MAX_TURNS : MAX_VAL_TURNS ))
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=2048
MAX_FEEDBACK_LENGTH=2048
TRAIN_BATCH_SIZE=1024
MINI_BATCH_SIZE=1024
MICRO_BATCH_SIZE_PER_GPU=1
TOTAL_LENGTH=$(($MAX_PROMPT_LENGTH+($MAX_RESPONSE_LENGTH+$MAX_FEEDBACK_LENGTH)*$TURNS_MAXIMUM))

python -m src.core.main_ppo \
    +env.name=$ENV \
    +env.search_url="http://127.0.0.1:8000/retrieve" \
    +env.search_topk=3 \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    +data.log_dir=$LOG_FILE_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$TOTAL_LENGTH \
    data.truncation=left \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    +data.max_feedback_length=$MAX_FEEDBACK_LENGTH \
    +data.max_turns=$MAX_TURNS \
    +data.val_max_turns=$MAX_VAL_TURNS \
    data.shuffle=false \
    +data.reward_allocate_method=step \
    +debug.check_token_align=false \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$N_NODES \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=verl_checkpoints/$EXP_NAME \
    trainer.logger="['$LOGGER']" \
    +trainer.val_only=false \
    trainer.val_before_train=true \
    trainer.critic_warmup=7 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 \
    +algorithm.paraphrase.enable=true \
    +algorithm.paraphrase.use_ref_policy=true \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.use_kl_in_reward=true \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.max_model_len=$TOTAL_LENGTH \
    actor_rollout_ref.rollout.max_num_batched_tokens=$TOTAL_LENGTH \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    +actor_rollout_ref.actor.sft_weight=1.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    +actor_rollout_ref.rollout.n_agent=$N_AGENTS \
    actor_rollout_ref.rollout.temperature=0.35 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    critic.model.path=$MODEL_PATH \
    critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps_ratio=0.0 \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    critic.model.use_remove_padding=True \
    2>&1 | tee $LOG_FILE_PATH/$EXP_NAME.log