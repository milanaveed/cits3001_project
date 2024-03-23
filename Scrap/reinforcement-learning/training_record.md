# 1-1
- Resized and simplified actions([Right] & [Right, A]) 
- `1-1_model_{}`
```
callback = TrainAndSaveCallback(check_freq=1000000, save_path=DATA_DIR)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR,
            learning_rate=0.000001, n_steps=512)
model.learn(8000000, callback=callback)
model.save('1-1-model')
```

# 1-2 SIMPLE-MOVEMENT NO-RESIZED 0.000001
- DATA_DIR = './reinforcement-learning/train/'
- model.save('1-2-model')

# 1-2 SIMPLE-MOVEMENT NO-RESIZED 0.0001
- DATA_DIR = './reinforcement-learning/train1-2-01/'
- model.save('1-2-01model')

<BR/>
Above are from version 1.

# 1-2 THREE-MOVEMENT RESIZED SKIPPEDFRAME 0.0001 CUSTOMIZED REWARD
- 11:34PM 13/10
- DATA_DIR = './reinforcement-learning/train1-2-02/'
- model.save('1-2-02model')

# 1-1 v3 
- DATA_DIR = './reinforcement-learning/train1-1-v302/'
- 3pm 14/10

# 1-2 v3
- DATA_DIR = './reinforcement-learning/train1-2-v3/'