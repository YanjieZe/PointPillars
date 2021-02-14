# Point Pillars Implementation

## make pillars算法
```
输入：包含n个4维点的点云
输出：包含n个9维点的点云
1. 创建一个pillars的字典，key为center，value为所包含的点的list，初始化为空.
2. 对于点云中的点，进行遍历。
（1）判断点是否在范围内。如果是，=>（2），否则跳过进入下一个点。
（2）判断点在哪个pillar内，加入对应pillar的list。
3. 对于已经创建好的pillars的字典内容进行遍历。
（1）如果该list的点含量大于100，随机采样其中的100个点，保留下来。如果该list的点含量小于100，用0填充至100。如果该list的点含量等于100，进入（2）。
（2）对于已经处理好的包含100个点的list进行遍历，将每个点由4维扩展为9维。
（3）将一个list转化为一个numpy矩阵。
4. 将字典中的所有numpy矩阵转化为一个numpy矩阵，输出。
    
```

## pybind编译指令
> c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) point_pillars.cpp -o point_pillars$(python3-config --extension-suffix)

## log
**2021.2.13**:pillar feature net finished.

**2021.2.14**:backbone finished.