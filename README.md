# Implementation of Reinforcement Learning algorithms on the Highway-Environment

### *CS525: Reinforcement Learning - [Worcester Polytechnic Institute](https://www.wpi.edu/), Fall 2022*

### Members: [Rutwik Bonde](https://github.com/Rubo12345), [Prathamesh Bhamare](https://github.com/Prathamesh1411), [Aniket Patil](https://github.com/aniketmpatil)

Master of Science in Robotics Engineering

#### [Link to Project Report](./Reinforcement_Learning_Course_Project.pdf)

---

## Setting up the Environment:

1. Create an empty python environment:
```
python3 -m venv rl_venv
```

2. Activate the environment and install libraries
```
source rl_venv/bin/activate
```
```
pip3 install -r requirements.txt
```

## Training:

### A3C Training:

python3 ./train.py --agent A3C --exp_id a3c1 --num_episodes 5000 --batch_size 256 --epsilon 0.6 --min_epsilon 0 --lr 0.00005 --lr_decay --arch Identity --fc_layers 3 --spawn_vehicles 3