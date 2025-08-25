from PIL import Image
import torch
from torch.utils import data 
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import streamlit as st
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 集成模型定义
class CombinedModel(nn.Module):  
    ''' 集成模型 '''
    def __init__(self, num_classes):  
        super(CombinedModel, self).__init__()  
             
        # ResNet Backbone  
        self.resnet = torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  
        num_ftrs_resnet = self.resnet.fc.in_features  
        self.resnet.fc = nn.Linear(num_ftrs_resnet, num_classes) 
        
        # EfficientNet Backbone  
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        num_ftrs_efficientnet = self.efficientnet._fc.in_features  
        self.efficientnet._fc = nn.Linear(num_ftrs_efficientnet, num_classes)  
        
        # Densenet Backbone
        self.densenet = torchvision.models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)  
        num_ftrs_densenet = self.densenet.classifier.in_features  
        self.densenet.classifier = nn.Linear(num_ftrs_densenet, num_classes) 
        
    def forward(self, x):  
        result_resnet = self.resnet(x)
        result_efficientnet = self.efficientnet(x)
        result_densenet = self.densenet(x)      
        output = (result_resnet + result_densenet + result_efficientnet ) / 3
        return output, result_resnet, result_efficientnet, result_densenet

# 为每个子模型定义目标层
def get_individual_target_layers(model):
    return {
        'resnet': [model.resnet.layer4[-1]],  # ResNet的目标卷积层
        'efficientnet': [model.efficientnet._blocks[-1]],  # EfficientNet的目标卷积层
        'densenet': [model.densenet.features[-1]]  # DenseNet的目标卷积层
    }

# 初始化模型
num_classes = 2  # 分类数 
model = CombinedModel(num_classes) 

# 加载训练好的模型
def load_model():
    try:
        model.load_state_dict(torch.load('./EDLM.pth', 
                                        map_location=torch.device('cpu')))
        model.eval()  # 设置为评估模式
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

model = load_model()
        
# 创建应用界面
st.title("Classification of Renal Fibrosis with SWE Image")
st.markdown("""
<style>
    .big-font {
        font-size:24px !important;
        color: #FF4B4B;
        font-weight: bold;
    }
    .medium-font {
        font-size:18px !important;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True) 
st.write("Welcome to intelligent renal fibrosis assessment.")

# 让用户上传图片
uploaded_file = st.file_uploader("Choose one SWE image (jpg)...", type="jpg")
if uploaded_file is not None and model is not None:
    # 显示上传的图片
    image = Image.open(uploaded_file).convert('RGB')  # 确保是RGB格式
    st.image(image, caption='Uploaded Image.', width=300, use_column_width=False)
    
    # 对图片进行预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)  # 添加batch维度

    # 使用模型进行预测
    with torch.no_grad():
        output, _, _, _ = model(input_tensor)  # 只获取集成模型输出，不需要子模型输出用于加权
        _, predicted_class = torch.max(output, 1)
        probabilities = F.softmax(output, dim=1)
        

    # 定义类别映射   
    class_mapping = {0: "mild", 1: "moderate-severe"}

    # 显示预测结果
    predicted_label = class_mapping[predicted_class.item()]
    with st.container():
        st.markdown(f'<p class="big-font">The predicted outcome is: {predicted_label}</p>', 
                    unsafe_allow_html=True)
        
        # 可视化预测概率的饼图
        st.write("### Classification probabilities Visualization")  
        labels = list(class_mapping.values())
        sizes = [probabilities[0][i].item() for i in range(len(class_mapping))]
        explode = [0.1 if i == predicted_class.item() else 0 for i in range(len(class_mapping))]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            sizes, 
            explode=explode, 
            labels=labels, 
            autopct='%1.2f%%',
            startangle=90,
            colors=['#66b3ff', '#ff9999']
        )
        plt.setp(autotexts, size=12)
        plt.setp(texts, size=12)
        ax.axis('equal')  
        st.pyplot(fig)

    # 生成各子模型的热力图并计算平均值
    st.write("### Model Interpretation with Grad-CAM")
    
    # 准备原始图像用于叠加
    img_np = np.array(image.resize((256, 256))) / 255.0  
    
    # 获取各模型的目标层
    target_layers = get_individual_target_layers(model)
    cams = []  # 存储各子模型的热力图
    
    # 为每个子模型生成热力图（不显示）
    for model_name in ['resnet', 'efficientnet', 'densenet']:
        try:
            # 创建GradCAM实例
            cam = GradCAM(
                model=getattr(model, model_name),
                target_layers=target_layers[model_name]
            )
            
            # 生成热力图
            targets = [ClassifierOutputTarget(predicted_class.item())]
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
                eigen_smooth=True,
                aug_smooth=True
            )
            grayscale_cam = grayscale_cam[0, :]  # 取第一个样本的热力图
            cams.append(grayscale_cam)
        except Exception as e:
            st.error(f"生成{model_name}热力图时出错: {str(e)}")
    
    # 计算平均热力图
    if len(cams) == 3:  # 确保成功获取了所有三个子模型的热力图
        # 简单平均
        avg_cam = np.mean(cams, axis=0)
              
        # 显示加权平均热力图
        st.subheader("Grad-CAM Heatmap")
        avg_visualization = show_cam_on_image(img_np, avg_cam, use_rgb=True)
        st.image(avg_visualization, width=300, use_column_width=False)
    else:
        st.warning("The heatmap could not be calculated")
