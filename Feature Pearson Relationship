import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

INPUT_FILE_PATH = r'C:\Users\Liu\Desktop\FS-YS-over.xlsx'
OUTPUT_IMAGE_PATH = r'C:\Users\Liu\Desktop\性能-特征相关性.png'

def generate_correlation_heatmap(input_path, output_image_path):
    try:
        df = pd.read_excel(input_path)
        print(f"读取成功，数据形状: {df.shape}")
        
        feature_columns = df.columns[:-2]
        target_columns = df.columns[-2:]
        
        print("\n特征列:")
        print(feature_columns.tolist())
        print("\n目标列（已排除）:")
        print(target_columns.tolist())
        
        features_df = df[feature_columns]
        correlation_matrix = features_df.corr(method='pearson')
        print("\n特征列相关系数矩阵计算完成")
        
        plt.rcParams["font.family"] = ["Times New Roman", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        
        mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=-1)
        
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                   linewidths=0, annot_kws={"size": 6}, mask=mask)
        
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        
        plt.xticks(fontsize=10, fontname='Times New Roman',rotation=45)
        plt.yticks(fontsize=10, fontname='Times New Roman')
        
        plt.title('特征相关性热图（保留对角线）', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"热图已保存到: {output_image_path}")
        
        plt.show()
        return correlation_matrix
        
    except Exception as e:
        print(f"处理出错: {str(e)}")
        return None

if __name__ == "__main__":
    correlation_matrix = generate_correlation_heatmap(INPUT_FILE_PATH, OUTPUT_IMAGE_PATH)
    
    if correlation_matrix is not None:
        print("\n特征相关矩阵预览:")
        print(correlation_matrix.head())
