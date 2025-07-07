import folium
import numpy as np
import pandas as pd
from itertools import pairwise
from typing import List, Tuple


# HSL 色相 H (0-360), 饱和度 S (0-100), 亮度 L (0-100) 转为十六进制颜色码
def hsl_to_hex(h: float, s: float, l: float) -> str:
    l /= 100
    a = s * min(l, 1 - l) / 100
    
    def f(n: int) -> str:
        k = (n + h / 30) % 12
        color = l - a * max(min(k - 3, 9 - k, 1), -1)
        return f'{int(round(255 * color)):02x}'
    
    return f'#{f(0)}{f(8)}{f(4)}'


# 根据标签的索引位置生成颜色
def generate_color_for_label(index: int) -> str:
    hue = (index * 137.508) % 360
    
    # 饱和度在区间内交替变化
    saturation_min = 65
    saturation_max = 90
    saturation = saturation_min if index % 2 == 0 else saturation_max
    
    # 亮度也在区间内交替变化，与饱和度交错
    lightness_min = 35
    lightness_max = 60
    # 当饱和度高时亮度低，饱和度低时亮度高，增强对比
    lightness = lightness_max if index % 2 == 0 else lightness_min
    
    return hsl_to_hex(hue, saturation, lightness)


def draw_map(points: np.ndarray, labels: np.ndarray, warehouse_coords: List[Tuple[float, float]] = None):
    m = folium.Map(location=np.mean(points, axis=0), zoom_start=9)
    for i, label in enumerate(np.unique(labels)):
        color = generate_color_for_label(i)
        cluster_points = points[labels == label]

        # 画点
        for j in range(len(cluster_points)):
            lat, lon = cluster_points[j]
            folium.CircleMarker(
                location=(lat, lon),
                radius=5,
                opacity=0,
                fill=True,
                fill_color=color,
                fill_opacity=1,
                tooltip=f'{label}'
            ).add_to(m)
    
    # 画仓库 - 在最后画，这样不会被其他点覆盖
    if warehouse_coords:
        for i, (lat, lon) in enumerate(warehouse_coords):
            folium.Marker(
                location=(lat, lon),
                popup=f'仓库 {i+1}',
                tooltip=f'仓库 {i+1}',
                icon=folium.Icon(color='red', icon='home', prefix='fa')
            ).add_to(m)
    
    return m
