import streamlit as st
import numpy as np
import random
import time
import tracemalloc
import math
from collections import deque
import heapq
import plotly.graph_objects as go

# 1. إعدادات الصفحة التفاعلية بمظهر احترافي ومريح للعين
st.set_page_config(page_title="AI Maze Solver Pro", layout="wide", initial_sidebar_state="collapsed")

# تصميم ستايل خاص بالصفحة لجعلها تبدو كلوحة تحكم فاخرة
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    h1 { color: #38bdf8; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("نجلاء الطويل و الاء الجزيري نبي نقوللكم انه اموركم طيبة احلا تحية للأنيق ")
st.markdown("<h4 style='text-align: center; color: #cbd5e1;'>مشروع مادة الذكاء الاصطناعي</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 15px;'>إعداد الطالبات: هبة محمود ابوسريويل & فاطمة خليفة | تحت إشراف: د. صلاح النعاس</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# كلاس توليد المتاهة
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

# دالات المسافة المساعدة
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
    return None, explored

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
    return None, explored

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
    return None, explored

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
    return None, explored

# ==========================================
# دالة الرسم (تم إصلاحها بالكامل لتحديث الألوان بشكل صحيح)
# ==========================================
def draw_maze(maze, explored, path=None):
    # إنشاء مصفوفة جديدة نظيفة في كل مرة لمنع تداخل الرسومات القديمة
    display_grid = np.copy(maze.grid).astype(float)
    
    # تلوين الخلايا المستكشفة (أزرق)
    if explored:
        for r, c in explored: 
            display_grid[r, c] = 0.4
            
    # تلوين مسار الحل الفعلي (برتقالي ذهبي)
    if path:
        for r, c in path: 
            display_grid[r, c] = 0.7
            
    # تحديد نقطة البداية والنهاية بدقة وثبات
    display_grid[maze.start[0], maze.start[1]] = 0.2
    display_grid[maze.end[0], maze.end[1]] = 0.9

    fig = go.Figure(data=go.Heatmap(
        z=display_grid,
        colorscale=[
            [0.0, '#1e293b'],    # الممرات (رمادي داكن)
            [0.2, '#10b981'],    # البداية (أخضر فوسفوري نيون)
            [0.4, '#1e40af'],    # العقد المستكشفة (أزرق ملكي داكن)
            [0.7, '#f59e0b'],    # مسار الحل الذهبي (برتقالي نيون مضيء)
            [0.9, '#ef4444'],    # النهاية (أحمر نيون)
            [1.0, '#020617']     # الجدران (أسود فاخر جداً)
        ],
        showscale=False, xgap=1, ygap=1
    ))
    fig.update_layout(
        yaxis=dict(autorange="reversed", showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=5, r=5, t=5, b=5),
        height=550, paper_bgcolor='#0f172a', plot_bgcolor='#0f172a'
    )
    return fig

# ==========================================
# بناء واجهة التحكم الرئيسية والتبديل السريع المضمون
# ==========================================
c_set1, c_set2 = st.columns([1, 1])
with c_set1:
    size_option = st.selectbox("📏 حجم المتاهة:", ["11x11 (صغير وسريع)", "31x31 (متوسط)", "51x51 (كبير)"])
    size_dict = {"11x11 (صغير وسريع)": 11, "31x31 (متوسط)": 31, "51x51 (كبير)": 51}
    grid_size = size_dict[size_option]
with c_set2:
    st.write("<br>", unsafe_allow_html=True)
    generate_btn = st.button("🔄 توليد متاهة عشوائية جديدة بالكامل", use_container_width=True)

# إدارة حالة المتاهة في الذاكرة
if "maze" not in st.session_state or generate_btn or st.session_state.get("last_size") != grid_size:
    st.session_state.maze = Maze(grid_size, grid_size)
    st.session_state.last_size = grid_size

current_maze = st.session_state.maze

st.markdown("---")
st.markdown("### 🗺️ اختر خوارزمية البحث للمقارنة الفورية:")

# استخدام Selectbox لمنع مشاكل الكاش والثبات في الرسم البياني
algo_choice = st.selectbox("اضغطي هنا لاختيار الخوارزمية ورؤية فرق الألوان والمسار الفعلي:", [
    "BFS (Breadth-First Search)",
    "DFS (Depth-First Search)",
    "Greedy (Manhattan)",
    "Greedy (Euclidean)",
    "A* (Manhattan)",
    "A* (Euclidean)"
])

algo_map_final = {
    "BFS (Breadth-First Search)": lambda m: bfs(m),
    "DFS (Depth-First Search)": lambda m: dfs(m),
    "Greedy (Manhattan)": lambda m: greedy_bfs(m, manhattan_distance),
    "Greedy (Euclidean)": lambda m: greedy_bfs(m, euclidean_distance),
    "A* (Manhattan)": lambda m: a_star(m, manhattan_distance),
    "A* (Euclidean)": lambda m: a_star(m, euclidean_distance)
}

# تشغيل الخوارزمية المختارة
algo_func = algo_map_final[algo_choice]

tracemalloc.start()
start_time = time.perf_counter()

current_path, current_explored = algo_func(current_maze)

end_time = time.perf_counter()
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

exec_time = (end_time - start_time) * 1000
mem_kb = peak_mem / 1024

# عرض الكروت الإحصائية الذكية اللي تتغير مع كل خوارزمية
m1, m2, m3, m4 = st.columns(4)
m1.metric("📏 طول مسار الحل", f"{len(current_path)} خطوة" if current_path else "لا يوجد حل")
m2.metric("🔍 العقد المستكشفة", f"{len(current_explored)} عقدة")
m3.metric("⚡ زمن التنفيذ", f"{exec_time:.3f} مللي ثانية")
m4.metric("💾 استهلاك الذاكرة", f"{mem_kb:.2f} KB")

# رسم المتاهة بالبيانات الفريدة للخوارزمية الحالية
fig = draw_maze(current_maze, current_explored, current_path)
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")
st.markdown("<div style='background-color: #1e293b; padding: 15px; border-radius: 10px; border-right: 5px solid #38bdf8; text-align: right;'>"
            "<span style='color: #38bdf8; font-weight: bold;'>💡 دليل قراءة الألوان الاحترافي:</span><br>"
            "<span style='color: #cbd5e1; font-size: 14px;'>"
            "• 🟥 المربع الأحمر: نقطة الهدف (End Node) | • 🟩 المربع الأخضر: نقطة انطلاق الذكاء الاصطناعي (Start Node)<br>"
            "• 🟦 المساحة الزرقاء الداكنة: تمثل كافة المسارات التي قامت الخوارزمية بفحصها واختبارها (Explored Area)<br>"
            "• 🟨 الخط البرتقالي الذهبي: يمثل المسار المثالي والأقصر الذي نجحت الخوارزمية في صياغته للوصول للهدف."
            "</span></div>", unsafe_allow_html=True)
