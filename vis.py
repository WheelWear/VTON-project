import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 데이터 준비
data = pd.DataFrame({
    'Year': [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030],
    'Market Size (Billion USD)': [91.7, 115.2, 142.5, 185.3, 228.9, 289.6, 365.8, 464.2]
})

# Seaborn 스타일 설정
sns.set_style("white")
sns.set_context("notebook", font_scale=1.2)

# 그래프 설정
plt.figure(figsize=(12, 8))

# Seaborn 산점도 그리기 (범례 제거)
scatter = sns.scatterplot(data=data, 
                          x='Year', 
                          y='Market Size (Billion USD)', 
                          size='Market Size (Billion USD)', 
                          sizes=(data['Market Size (Billion USD)'].min() * 20, data['Market Size (Billion USD)'].max() * 40), 
                          hue='Market Size (Billion USD)', 
                          palette='viridis', 
                          alpha=0.7, 
                          edgecolor='black', 
                          linewidth=1, 
                          legend=False)  # 크기 및 색상 범례 제거

# 색상 바 추가 (시장 규모 값 기준)
cbar = plt.colorbar(scatter.collections[0], label='Market Size (Billion USD)')
cbar.set_ticks([100, 200, 300, 400, 500])  # 색상 바 눈금 명시적으로 설정

# 그래프 꾸미기
plt.title('Market Size by Year (Billion USD)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Market Size (Billion USD)', fontsize=14)

# 각 데이터 포인트에 값 표시
for i, row in data.iterrows():
    plt.text(row['Year'], row['Market Size (Billion USD)'], f"{row['Market Size (Billion USD)']}", 
             ha='center', va='center', 
             fontsize=10, fontweight='bold', 
             color='white' if row['Market Size (Billion USD)'] > 200 else 'black')

# 축 범위 조정
plt.xlim(2022, 2031)
plt.ylim(0, 600)

# 배경 설정
plt.gca().set_facecolor('#f9f9f9')
plt.tight_layout()

# 그래프 표시
plt.show()