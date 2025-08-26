# PlayerImputer

`PlayerImputer` is a model designed to predict and impute player positions based on soccer match data. Currently, it supports training and testing using a transformer-based model.

---

## 1. Environment Setup

- **Python version**: 3.9.21  
- **Dependencies**: Install via `requirements.txt`
```bash
pip install -r requirements.txt
````

---

## 2. Project Structure

* 📂`data` : Training and test datasets
* 📂`datatools` : Data preprocessing utilities
* 📂`imputer` : Model implementation
* 📂`notebook` : Analysis and experiment notebooks
* 📂`stores` : Trained models and results
* 📂`socceraction` : Soccer event processing module
* `imputer_params.json` : Model hyperparameters
* `train.py` : Training and testing script
* `train_model.sh` : Script for automating training
---

## 3. Training

Run the following command to train the model:

```bash
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
--trial 0 \
--data_version 0
```

---

## 4. Testing

To test a trained model, use:

```bash
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
--trial 0 \
--data_version 0
```

---


## 5. Notes

* A tutorial will be added in future updates.
* Ensure the experimental environment and data version match to reproduce the results exactly.

---



