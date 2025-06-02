import random
import folium
import numpy as np
import pandas as pd
from itertools import pairwise


# 生成随机颜色值
def random_color():
    while True:
        r = random.randint(0, 0xff)
        g = random.randint(0, 0xff)
        b = random.randint(0, 0xff)
        if max(r, g, b) - min(r, g, b) > 0x88:
            return f'#{r:02x}{g:02x}{b:02x}'


def draw_map(points: np.ndarray, labels: np.ndarray):
    m = folium.Map(location=np.mean(points, axis=0), zoom_start=9)
    for label in np.unique(labels):
        color = random_color()
        cluster_points = points[labels == label]

        # 画点
        for i in range(len(cluster_points)):
            lat, lon = cluster_points[i]
            folium.CircleMarker(
                location=(lat, lon),
                radius=5,
                opacity=0,
                fill=True,
                fill_color=color,
                fill_opacity=1,
                tooltip=f'{label}'
            ).add_to(m)
    return m
