# Policy-Gradient-Lunarlander
The best result of LunarLander-v2 using deep reinforcement learning(DRL) algorithm Policy Gradient reinforce I could ever try.

By testing 200 times. 189 times reward > 200.(Test reward>=200:(189/200). Avg steps:285. Avg reward:  248.78)

weight file is:
./w/w_best.pt_rw270_lr0.013000_gm0.9880_ep3000

change TEST_MODE to switch between training and testing

the different thing I have done to the code is:
1. limit the TIMEOUT_STEP to 390
2. set the finial reward to -100.0 when it is time out.

Without using LR decay(decay rate is 1), the result is not converge steadily, however, it produce good result accidentally.
