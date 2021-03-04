1.script:
script中有两个文件：
function_solve_dp.py中新增了四个函数：
solve_dp_noturn_valueiteration
solve_dp_noturn_policyiteration
solve_dp_turn_valueiteration
solve_dp_turn_policyiteration（加分项）
分别对应考虑turn、不考虑turn 使用value iteration和policy iteration的情况

runscript_solve_dp_all.py中
可以写好了数据接口，和调用函数的代码，可以直接运行得到结果
这两个代码文件都需要放置在script文件夹下

2.result
-----不考虑turn-----
直接解
singlegame_player7_noturn_partactionset.pkl
singlegame_player7_noturn_fullactionset.pkl
迭代解
Value iteration
singlegame_player7_noturn_partactionset_policyiter.pkl:
Policy iteration
singlegame_player7_noturn_partactionset_valueiter.pkl

------考虑turn-----
1.value iteration
On part of aiming set
singlegame_player7_turn_partactionset_valueiter.pkl
On full aiming set
singlegame_player7_turn_fullactionset_valueiter.pkl
2.policy iteration
On part of aiming set
singlegame_player7_turn_partactionset_policyiter.pkl

3.report
Dart_Project-final.pdf
