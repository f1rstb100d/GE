1. 创建虚拟环境
```
conda create -n stellarg python=3.7 
```
2. 删除虚拟环境
```
conda remove -n stellarg --all
```
3. 激活虚拟环境（第一步可省略）
```
source activate stellarg
conda activate stellarg
```
4. 查看环境下已有的包
```
conda list
```
5. 虚拟环境下安装包
```
pip install stellargraph
conda install stellargraph
```
6. 退出当前虚拟环境
```
conda deactivate
```
7. 查看已有虚拟环境
```
conda-env list
```


conda install提示在当前的channels中找不到这个包
```
anaconda search -t conda stellargraph
anaconda show stellargraph/stellargraph
conda install --channel https://conda.anaconda.org/stellargraph stellargraph
```