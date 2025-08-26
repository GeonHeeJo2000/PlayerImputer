python train.py \
--data_dir data/DFL \
--model transformer \
--window 3 \
--xfns "prevAgentX" "prevAgentY" "prevAgentTime" "nextAgentX" "nextAgentY" "nextAgentTime" "avgAgentX" "avgAgentY" "agentSide" "agentRole" "agentObserved" "goalDiff" "eventX" "eventY" "eventType" "freeze_frame" "observeEventX" "observeEventY" "prevAvgX" "prevAvgY" "nextAvgX" "nextAvgY" "possessRatio" "elapsedTime" \
--yfns "coordinates" \
--use_transform \
--play_left_to_right \
--params_file imputer_params.json \
--trial 0

python train.py \
--data_dir data/DFL \
--model transformer \
--window 3 \
--xfns "prevAgentX" \
--yfns "coordinates" \
--use_transform \
--play_left_to_right \
--params_file imputer_params.json \
--trial 0

python train.py \
--data_dir data/DFL \
--model transformer \
--window 3 \
--xfns "prevAgentX" \
--yfns "coordinates" \
--use_transform \
--play_left_to_right \
--params_file imputer_params.json \
--trial 0

python train.py \
--data_dir data/DFL \
--mode train \
--model transformer \
--window 11 \
--xfns "prevAgentX" "prevAgentY" "prevAgentTime" "nextAgentX" "nextAgentY" "nextAgentTime" "avgAgentX" "avgAgentY" "agentSide" "agentRole" "agentObserved" "goalDiff" "eventX" "eventY" "eventType" "freeze_frame" "observeEventX" "observeEventY" "prevAvgX" "prevAvgY" "nextAvgX" "nextAvgY" "possessRatio" "elapsedTime" \
--yfns "coordinates" \
--use_transform \
--play_left_to_right \
--params_file imputer_params.json \
--trial 27

 python train.py \
--data_dir data/DFL \
--mode test \
--model transformer \
--window 41 \
--xfns "prevAgentX" "prevAgentY" "prevAgentTime" "nextAgentX" "nextAgentY" "nextAgentTime" "avgAgentX" "avgAgentY" "agentSide" "agentRole" "agentObserved" "goalDiff" "eventX" "eventY" "eventType" "freeze_frame" "observeEventX" "observeEventY" "prevAvgX" "prevAvgY" "nextAvgX" "nextAvgY" "possessRatio" "elapsedTime" \
--yfns "coordinates" \
--use_transform \
--play_left_to_right \
--params_file imputer_params.json \
--trial 50 \
--data_version 47

conda activate playerimputer
 python train.py \
--data_dir data/DFL \
--mode train \
--model transformer \
--window 41 \
--xfns "prevAgentX" "prevAgentY" "prevAgentTime" "nextAgentX" "nextAgentY" "nextAgentTime" "avgAgentX" "avgAgentY" "agentSide" "agentRole" "agentObserved" "goalDiff" "eventX" "eventY" "eventType" "observeEventX" "observeEventY" "prevAvgX" "prevAvgY" "nextAvgX" "nextAvgY" "possessRatio" "elapsedTime" \
--yfns "coordinates" \
--use_transform \
--play_left_to_right \
--params_file imputer_params.json \
--trial 154 \
--data_version 154

tmux new -s version44
conda activate playerimputer
 python train.py \
--data_dir data/DFL \
--mode test \
--model transformer \
--window 41 \
--xfns "freeze_frame" "prevAgentX" "prevAgentY" "prevAgentTime" "nextAgentX" "nextAgentY" "nextAgentTime" "avgAgentX" "avgAgentY" "agentSide" "agentRole" "agentObserved" "goalDiff" "eventX" "eventY" "eventType" "observeEventX" "observeEventY" "prevAvgX" "prevAvgY" "nextAvgX" "nextAvgY" "possessRatio" "elapsedTime" \
--yfns "coordinates" \
--use_transform \
--play_left_to_right \
--params_file imputer_params.json \
--trial 48 \
--data_version 46

tmux new -s version12
conda activate playerimputer