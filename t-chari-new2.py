import os
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np

# 加载预训练的GAN模型
gan_model = hub.load('https://tfhub.dev/deepmind/biggan-deep-256/1')  # BigGAN-deep-256模型

# 指定保存生成图像的目录
output_dir = os.path.expanduser("~/Desktop/generated_images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def generate_initial_image():
    z = tf.random.normal([1, 128])  # BigGAN需要128维的噪声向量
    
    # 指定椅子类别，对于BigGAN模型，椅子的类别标签是62
    y = tf.one_hot([985], depth=5000)  
    
    # 生成初始图像
    model_output = gan_model.signatures['default'](truncation=tf.constant(0.5), z=tf.constant(z), y=tf.constant(y))
    
    # 获取生成的图像并归一化像素值到0-1的范围
    generated_image = (model_output['default'][0] + 1) / 2.0
    
    return generated_image, z

def save_image(image, epoch):
    # 保存图像
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'image_{epoch}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

def loop_generate_and_influence(initial_image, initial_z, iterations):
    for i in range(iterations):
        # 显示和保存当前生成的图像
        save_image(initial_image, i)
        
        # 通过对噪声向量进行小的随机调整来影响下一张图像的生成
        new_z = initial_z + tf.random.normal(initial_z.shape, mean=0.0, stddev=0.05)
        
        # 生成新的图像
        y = tf.one_hot([985], depth=1000)  # 椅子的类别标签
        model_output = gan_model.signatures['default'](truncation=tf.constant(0.5), z=new_z, y=y)
        
        # 获取生成的图像并归一化像素值到0-1的范围
        initial_image = (model_output['default'][0] + 1) / 2.0
        initial_z = new_z

# 生成初始图像
initial_image, initial_z = generate_initial_image()

# 启动循环来生成和保存多张图像
loop_generate_and_influence(initial_image, initial_z, iterations=10000)
