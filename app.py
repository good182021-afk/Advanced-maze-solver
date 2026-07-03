import streamlit as st
import numpy as np
import random
import time
import tracemalloc
import math
from collections import deque
import heapq
import plotly.graph_objects as go
import pandas as pd

# 1. إعدادات الصفحة التفاعلية بعرض الشاشة الكاملة
st.set_page_config(page_title="AI Maze Solver Pro", layout="wide", initial_sidebar_state="collapsed")

# تنسيق المظهر العام لإعطاء لمسة داشبورد فاخرة
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    h1, h2, h3 { color: #38bdf8; font-family: 'Segoe UI', system-ui, sans-serif; text-align: center; }
    .stButton>button { background-color: #0284c7; color: white; border-radius: 8px; font-weight: bold; }
    .stButton>button:hover { background-color: #0369a1; border-color: #38bdf8; }
    [data-testid="stMetricValue"] { font-size: 20px !important; color: #f59e0b !important; }
    div[style*="flex-direction: row-reverse"] { direction: rtl; }
    </style>
""", unsafe_allow_html=True)

st.title("Advanced Maze Solver")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 15px;'>إعداد الطالبات: هبة محمود ابوسريويل & فاطمة خليفة | تحت إشراف: د. صلاح النعاس</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# كلاس وبنية المتاهة
# ==========================================
class Maze:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = np.ones((rows, cols), dtype=int)
        self.start = (0, 0)
        self.end = (rows - 1, cols - 1)
        self._generate_perfect_maze()
        
    def _get_neighbors(self, r, c):
        neighbors = []
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc, dr, dc))
        return neighbors

    def _generate_perfect_maze(self):
        stack = [(0, 0)]
        self.grid[0][0] = 0
        visited = {(0, 0)}

        while stack:
            r, c = stack[-1]
            neighbors = self._get_neighbors(r, c)
            unvisited_neighbors = [n for n in neighbors if (n[0], n[1]) not in visited]

            if unvisited_neighbors:
                nr, nc, dr, dc = random.choice(unvisited_neighbors)
                self.grid[r + dr//2][c + dc//2] = 0
                self.grid[nr][nc] = 0
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()
                
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.end[0]][self.end[1]] = 0
        self.grid[self.end[0]-1][self.end[1]] = 0
        self.grid[self.end[0]][self.end[1]-1] = 0

    def get_valid_neighbors(self, pos):
        r, c = pos
        results = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] == 0:
                results.append((nr, nc))
        return results

def manhattan_distance(p1, p2): return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
def euclidean_distance(p1, p2): return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def reconstruct_path(came_from, start, end):
    path = []
    current = end
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# ==========================================
# خوارزميات البحث الذكية
# ==========================================
def bfs(maze):
    start, end = maze.start, maze.end
    queue = deque([start])
    came_from = {start: None}
    explored = []
    explored_set = {start}
    while queue:
        current = queue.popleft()
        explored.append(current)
        if current == end: return reconstruct_path(came_from, start, end), explored
        for neighbor in maze.get_valid_neighbors(current):
            if neighbor not in explored_set:
                explored_set.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
    return [], explored

def dfs(maze):
    start, end = maze.start, maze.end
    stack = [start]
    came_from = {start: None}
    explored = []
    explored_set = set()
    while stack:
        current = stack.pop()
        if current in explored_set: continue
        explored_set.add(current)
        explored.append(current)
        if current == end: return reconstruct_path(came_from, start, end), explored
        for neighbor in maze.get_valid_neighbors(current):
            if neighbor not in explored_set:
                came_from[neighbor] = current
                stack.append(neighbor)
    return [], explored

def greedy_bfs(maze, heuristic_func):
    start, end = maze.start, maze.end
    open_set = [(heuristic_func(start, end), start)]
    came_from = {start: None}
    explored = []
    explored_set = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current in explored_set: continue
        explored_set.add(current)
        explored.append(current)
        if current == end: return reconstruct_path(came_from, start, end), explored
        for neighbor in maze.get_valid_neighbors(current):
            if neighbor not in explored_set and neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(open_set, (heuristic_func(neighbor, end), neighbor))
    return [], explored

def a_star(maze, heuristic_func):
    start, end = maze.start, maze.end
    open_set = [(heuristic_func(start, end), start)]
    came_from = {start: None}
    g_score = {start: 0}
    explored = []
    explored_set = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current in explored_set: continue
        explored_set.add(current)
        explored.append(current)
        if current == end: return reconstruct_path(came_from, start, end), explored
        for neighbor in maze.get_valid_neighbors(current):
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic_func(neighbor, end)
                came_from[neighbor] = current
                heapq.heappush(open_set, (f_score, neighbor))
    return [], explored

# ==========================================
# دالة الرسم المحسنة والمكبرة
# ==========================================
def draw_maze(maze, explored, path=None, title_text=""):
    display_grid = np.copy(maze.grid).astype(float)
    if explored:
        for r, c in explored: display_grid[r, c] = 0.4
    if path:
        for r, c in path: display_grid[r, c] = 0.7
    display_grid[maze.start[0], maze.start[1]] = 0.2
    display_grid[maze.end[0], maze.end[1]] = 0.9

    fig = go.Figure(data=go.Heatmap(
        z=display_grid,
        colorscale=[
            [0.0, '#1e293b'], [0.2, '#10b981'], [0.4, '#1e40af'],
            [0.7, '#f59e0b'], [0.9, '#ef4444'], [1.0, '#020617']
        ],
        showscale=False, xgap=1, ygap=1
    ))
    fig.update_layout(
        title=dict(text=title_text, x=0.5, y=0.95, font=dict(color="#38bdf8", size=16)),
        yaxis=dict(autorange="reversed", showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=40, b=10),
        height=580, paper_bgcolor='#0f172a', plot_bgcolor='#0f172a'
    )
    return fig

# ==========================================
# التوزيع الهيكلي الجديد للواجهة (يسار ويمين)
# ==========================================
col_left, col_right = st.columns([1, 2.2]) # تقسيم الشاشة: اليسار أصغر واليمين للمتاهة الكبيرة

# دالات تشغيل كل الخوارزميات لحساب الأداء والمقارنة الكلية
algo_dict_all = {
    "BFS": lambda m: bfs(m),
    "DFS": lambda m: dfs(m),
    "Greedy (Manhattan)": lambda m: greedy_bfs(m, manhattan_distance),
    "Greedy (Euclidean)": lambda m: greedy_bfs(m, euclidean_distance),
    "A* (Manhattan)": lambda m: a_star(m, manhattan_distance),
    "A* (Euclidean)": lambda m: a_star(m, euclidean_distance)
}

with col_left:
    st.markdown("### ⚙️ لوحة التحكم")
    size_option = st.selectbox("📏 حجم المتاهة:", ["11x11 (صغير وسريع)", "31x31 (متوسط)", "51x51 (كبير)"])
    size_dict = {"11x11 (صغير وسريع)": 11, "31x31 (متوسط)": 31, "51x51 (كبير)": 51}
    grid_size = size_dict[size_option]
    
    generate_btn = st.button("🔄 توليد متاهة جديدة", use_container_width=True)

    if "maze" not in st.session_state or generate_btn or st.session_state.get("last_size") != grid_size:
        st.session_state.maze = Maze(grid_size, grid_size)
        st.session_state.last_size = grid_size

    current_maze = st.session_state.maze

    st.markdown("---")
    st.markdown("### 🔍 خيارات العرض والبحث")
    algo_choice = st.selectbox("اختر الخوارزمية للعرض الفوري:", [
        "🔄 مقارنة جميع الخوارزميات معاً",
        "BFS (Breadth-First Search)",
        "DFS (Depth-First Search)",
        "Greedy (Manhattan)",
        "Greedy (Euclidean)",
        "A* (Manhattan)",
        "A* (Euclidean)"
    ])

# حساب نتائج كل الخوارزميات لتغذية المقارنة بالأسفل
all_results = {}
for name, func in algo_dict_all.items():
    tracemalloc.start()
    t_start = time.perf_counter()
    path_res, exp_res = func(current_maze)
    t_end = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    all_results[name] = {
        "path": path_res,
        "explored": exp_res,
        "steps": len(path_res) if path_res else 0,
        "nodes": len(exp_res),
        "time": (t_end - t_start) * 1000,
        "memory": peak_mem / 1024
    }

# العرض في العمود الأيمن (الرسمة الكبيرة)
with col_right:
    if algo_choice == "🔄 مقارنة جميع الخوارزميات معاً":
        # عند اختيار الكل، نعرض خوارزمية A* كنموذج مثالي على اليمين ونشرح للمستخدم
        fig = draw_maze(current_maze, all_results["A* (Manhattan)"]["explored"], all_results["A* (Manhattan)"]["path"], "رسم توضيحي لخوارزمية A* (النموذج الأقوى للحل)")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        res = all_results[algo_choice]
        fig = draw_maze(current_maze, res["explored"], res["path"], f"مسار استكشاف وحل خوارزمية: {algo_choice}")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
# ==========================================
# الجزء السفلي: نتائج المقارنة والتحليل الذكي الشامل
# ==========================================
st.markdown("---")
st.markdown("### 📊 جدول تحليل الأداء والمقارنة الشاملة بين الخوارزميات")

# تجهيز البيانات لعرضها في جدول تفاعلي منظم جداً للدكتور
data_list = []
for name, res in all_results.items():
    data_list.append({
        "الخوارزمية": name,
        "طول مسار الحل (خطوة)": res["steps"],
        "عدد العقد المستكشفة (عقدة)": res["nodes"],
        "زمن التنفيذ (مللي ثانية)": round(res["time"], 4),
        "استهلاك الذاكرة (KB)": round(res["memory"], 2)
    })

df = pd.DataFrame(data_list)
st.dataframe(df, use_container_width=True, hide_index=True)

# رسم بياني للمقارنة السريعة بين الأزمنة لزيادة قوة المناقشة
st.markdown("#### ⚡ مقارنة بيانية سريعة لزمن التنفيذ (كلما قلّ الشريط كان الأداء أفضل):")
chart_fig = go.Figure(data=[
    go.Bar(
        x=df["الخوارزمية"], 
        y=df["زمن التنفيذ (مللي ثانية)"],
        marker_color=['#38bdf8', '#818cf8', '#34d399', '#f43f5e', '#fbbf24', '#a78bfa']
    )
])
chart_fig.update_layout(
    paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
    font=dict(color='#cbd5e1'), margin=dict(l=20, r=20, t=20, b=20), height=250
)
st.plotly_chart(chart_fig, use_container_width=True, config={'displayModeBar': False})

st.markdown("<div style='background-color: #1e293b; padding: 15px; border-radius: 10px; border-right: 5px solid #38bdf8; text-align: right;'>"
            "<span style='color: #38bdf8; font-weight: bold;'>💡 دليل قراءة الألوان الاحترافي للمتاهة:</span><br>"
            "<span style='color: #cbd5e1; font-size: 14px;'>"
            "• 🟥 المربع الأحمر: نقطة الهدف (End Node) | • 🟩 المربع الأخضر: نقطة انطلاق الذكاء الاصطناعي (Start Node)<br>"
            "• 🟦 المساحة الزرقاء الداكنة: تمثل كافة المسارات التي قامت الخوارزمية بفحصها واختبارها (Explored Area)<br>"
            "• 🟨 الخط البرتقالي الذهبي: يمثل المسار المثالي والأقصر الذي نجحت الخوارزمية في صياغته للوصول للهدف."
            "</span></div>", unsafe_allow_html=True)
