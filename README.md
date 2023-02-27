### 运行环境

> 1.pytorch的cpu版本
>
> 2.python3.7
>
> 3.Windows

### 依赖包说明

> torch.nn
>
> xarray
>
> numpy

### 项目使用说明

主函数在ConvLSTM文件中，数据文件在Data文件夹中，过程数据在process文件夹中。每次先在read_file文件中执行示例函数，读取数据并处理后，再运行ConvLSTM文件。控制台会打印训练的轮数及损失。