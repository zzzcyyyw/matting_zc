# matting_zc

## 权重文件下载说明
由于save_model/adapter/train_adapter_221.pth和Depth_Anything_V2/checkpoints/depth_anything_v2_vitb.pth权重文件大于100M，需使用git lfs相关命令进行拉取。  
  
## 代码运行
整个代码运行分为了两个阶段：  
  
### stage1 训练depth adapter
深度图通过clip的图像编码器和编写的深度图适配器（depth adapter）转换为text embedding。  
1. 设置utils中的args.stage=1  
2. 运行 train_depth_adapter_prompt  

### stage2 训练解码器
将RGB图通过clip的图像编码器和stage1得到的text embedding一起送入解码器进行训练。  
1. 设置utils中的args.stage=2  
2. 运行 train_depth_adapter_prompt  
