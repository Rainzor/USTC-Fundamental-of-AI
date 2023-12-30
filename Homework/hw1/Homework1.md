# Homework1 

> Artificial Intelligence
>
> PB20020480 王润泽

### Q1

**3.7** Give the initial state, goal test, successor  function, and cost function for each of the following. Choose a  formulation that is precise enough to be implemented.
a. You have to color a planar map using only four colors, in such a way that no two adjacent regions have the same color.

b. A 3-foot-tall monkey is in a room where some bananas are suspended from the 8-foot ceiling. He would like to get the bananas. The room contains two stackable, movable, climbable 3-foot-high crates.

d. You have three jugs, measuring 12 gallons, 8 gallons, and 3 gallons, and a water faucet. You can fill the jugs up or empty  them out from one to another or onto the ground. You need to measure out exactly one gallon.  

### A1

a. 

初始状态：没有颜色的地图

目标测试：地图地区颜色涂满且两个相邻地区颜色不同

后继函数：用4种颜色的一种涂一个地图颜色后依然满足任意两个相邻地区颜色不同的地图状态

耗散函数：涂色次数

b.

初始状态：猴子和箱子任意处于某个位置

目标测试：箱子叠放，且猴子攀登后可以获取香蕉

后继函数：产生通过箱子移动、叠放、攀登后箱子的合法状态

耗散函数：每一步动作的耗散值为1，整个过程的耗散值为总步数

d.

初始状态: 三个水壶容量为12加仑，8加仑和3加仑，且所有水壶为空

目标测试：存在一个水壶内只有1加仑水

后继函数：经过某个水壶装满、倒空，或从一个水壶向另一个水壶倒水后的合法状态（即不存在水壶里的水超过容积）

耗散函数：每一步动作的耗散值为1，整个过程的耗散值为总步数

### Q2

**3.9** The **missionaries and cannibals**  problem is usually stated as follows. Three missionaries and three  cannibals are on one side of a river, along with a boat that can hold  one or two people. Find a way to get everyone to the other side without  ever leaving a group of missionaries in one place outnumbered by the  cannibals in that place. This problem is famous in AI because it was the subject of the first paper that approached problem formulation from an  analytical viewpoint (Amarel, 1968).

a. Formulate the problem  precisely, making only those distinctions necessary to ensure a valid  solution. Draw a diagram of the complete state space.

b. Implement and solve the problem optimally using an appropriate search algorithm.  Is it a good idea to check for repeated states?

c. Why do you think people have a hard time solving this puzzle, given that the state space is so simple?  

### A2

![](D:\ComputerScience\cs_2023_Spring_AI\Homework\hw1.2.png)