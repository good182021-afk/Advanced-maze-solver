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

# إعدادات الصفحة التفاعلية
st.set_page_config(page_title="Advanced Maze Solver AI Pro", layout="wide")
st.title(" 🧠 Advanced Maze Solver Using AI Search Algorithms (With Bonus Features)")
st.markdown("### مشروع مادة الذكاء الاصطناعي - إعداد: هبة محمود & فاطمة خليفة | تحت إشراف: د. صلاح النعاس")

# ==========================================
# 1. كلاس توليد المتاهة مع دعم التضاريس الموزونة (Weighted Terrain)
# ==========================================
class Maze:
    def __init__(self, rows, cols, add_terrain=False):
        self.rows = rows
        self.cols = cols
        # القيم: 0: ممر عادي (تكلفة 1)، 1: جدار، 3: رمل (تكلفة 3)، 5: طين (تكلفة 5)
        self.grid = np.ones((rows, cols), dtype=int)
        self.start = (0, 0)
        self.end = (rows - 1, cols - 1)
        self._generate_perfect_maze()
        if add_terrain:
            self._add_weighted_terrain()
        
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

    def _add_weighted_terrain(self):
        """إضافة تضاريس موزونة عشوائياً (رمل وطين) في الممرات السالكة"""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 0 and (r, c) != self.start and (r, c) != self.end:
                    rand_val = random.random()
                    if rand_val < 0.15:
                        self.grid[r][c] = 3 # رمل (تكلفة الحركة = 3)
                    elif rand_val < 0.25:
                        self.grid[r][c] = 5 # طين (تكلفة الحركة = 5)

    def get_cost(self, pos):
        """إرجاع تكلفة الحركة إلى الخلية"""
        r, c = pos
        val = self.grid[r][c]
        if val == 1:
            return float('inf') # جدار
        return val if val > 0 else 1 # الممر العادي تكلفته 1

    def get_valid_neighbors(self, pos):
        r, c = pos
        results = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 1:
                results.append((nr, nc))
        return results

# ==========================================
# 2. دالات المسافة والمسار
# ==========================================
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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
# 3. خوارزميات البحث الشاملة (تأخذ التكاليف الموزونة في الاعتبار)
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
        if current == end:
            return reconstruct_path(came_from, start, end), explored
            
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
        if current in explored_set:
            continue
        explored_set.add(current)
        explored.append(current)
        
        if current == end:
            return reconstruct_path(came_from, start, end), explored
            
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
        if current in explored_set:
            continue
        explored_set.add(current)
        explored.append(current)
        
        if current == end:
            return reconstruct_path(came_from, start, end), explored
            
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
        if current in explored_set:
            continue
        explored_set.add(current)
        explored.append(current)
        
        if current == end:
            return reconstruct_path(came_from, start, end), explored
            
        for neighbor in maze.get_valid_neighbors(current):
            move_cost = maze.get_cost(neighbor)
            tentative_g_score = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic_func(neighbor, end)
                came_from[neighbor] = current
                heapq.heappush(open_set, (f_score, neighbor))
    return None, explored

# 🌟 [BONUS FEATURE 1]: خوارزمية IDA* (Iterative Deepening A*)
def ida_star(maze, heuristic_func):
    start, end = maze.start, maze.end
    bound = heuristic_func(start, end)
    path = [start]
    explored = []
    explored_set = set()

    def search(path, g, bound):
        node = path[-1]
        if node not in explored_set:
            explored_set.add(node)
            explored.append(node)
            
        f = g + heuristic_func(node, end)
        if f > bound:
            return f, False
        if node == end:
            return f, True
            
        min_bound = float('inf')
        for neighbor in maze.get_valid_neighbors(node):
            if neighbor not in path:
                path.append(neighbor)
                cost = maze.get_cost(neighbor)
                t, found = search(path, g + cost, bound)
                if found:
                    return t, True
                if t < min_bound:
                    min_bound = t
                path.pop()
        return min_bound, False

    # حد أقصى للتكرارات لمنع التعليق في المتاهات الكبيرة جداً
    max_iterations = 100
    for _ in range(max_iterations):
        t, found = search(path, 0, bound)
        if found:
            return path, explored
        if t == float('inf'):
            return None, explored
        bound = t
        
    return None, explored

# ==========================================
# 4. دالة الرسم التفاعلي المطور الموزون
# ==========================================
def draw_maze(maze, explored, path=None):
    display_grid = np.copy(maze.grid).astype(float)
    
    # تحويل التضاريس إلى قيم نسبية للألوان
    # 0.0: ممر عادي | 0.3: رمل (تكلفة 3) | 0.5: طين (تكلفة 5)
    for r in range(maze.rows):
        for c in range(maze.cols):
            if display_grid[r, c] == 3:
                display_grid[r, c] = 0.3
            elif display_grid[r, c] == 5:
                display_grid[r, c] = 0.5
            elif display_grid[r, c] == 0:
                display_grid[r, c] = 0.0
                
    if explored:
        for r, c in explored:
            if (r, c) != maze.start and (r, c) != maze.end:
                display_grid[r, c] = 0.65 # تم استكشافه (أزرق)
            
    if path:
        for r, c in path:
            if (r, c) != maze.start and (r, c) != maze.end:
                display_grid[r, c] = 0.85 # مسار الحل الفعلي (ذهبي)
            
    display_grid[maze.start[0], maze.start[1]] = 0.15 # البداية (أخضر)
    display_grid[maze.end[0], maze.end[1]] = 1.0 # النهاية (أحمر)

    fig = go.Figure(data=go.Heatmap(
        z=display_grid,
        colorscale=[
            [0.0, '#ffffff'],    # الممرات العادية (أبيض)
            [0.15, '#10b981'],   # نقطة البداية (أخضر)
            [0.3, '#fde047'],    # 🌟 رمل (أصفر - تكلفة 3)
            [0.5, '#b45309'],    # 🌟 طين (بني - تكلفة 5)
            [0.65, '#60a5fa'],   # العقد المستكشفة (أزرق)
            [0.85, '#f59e0b'],   # مسار الحل (برتقالي ذهبي)
            [1.0, '#1e293b']     # الجدران (أسود/رمادي غامق)
        ],
        showscale=False,
        xgap=1, ygap=1
    ))
    
    fig.update_layout(
        yaxis=dict(autorange="reversed", showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=580,
        paper_bgcolor='#0f172a', plot_bgcolor='#0f172a'
    )
    return fig

# ==========================================
# 5. واجهة المستخدم والتفاعل
# ==========================================
col1, col2 = st.columns([1.1, 2])

with col1:
    st.sidebar.header("⚙️ إعدادات المحاكاة والبونص")
    
    size_option = st.sidebar.selectbox("📏 اختر حجم المتاهة:", ["11x11 (صغير)", "31x31 (متوسط)", "51x51 (كبير)"])
    size_dict = {"11x11 (صغير)": 11, "31x31 (متوسط)": 31, "51x51 (كبير)": 51}
    grid_size = size_dict[size_option]
    
    # 🌟 [BONUS FEATURE 2]: تفعيل التضاريس الموزونة
    enable_terrain = st.sidebar.checkbox("🏜️ تفعيل التضاريس الموزونة (Weighted Terrain - رمل وطين)")
    
    algo_name = st.sidebar.selectbox("اختر خوارزمية البحث:", [
        "BFS (Breadth-First Search)", 
        "DFS (Depth-First Search)", 
        "Greedy (Manhattan Distance)", 
        "Greedy (Euclidean Distance)",
        "A* (Manhattan Distance)",
        "A* (Euclidean Distance)",
        "🌟 IDA* (Iterative Deepening A* - Manhattan)"
    ])
    
    generate_btn = st.sidebar.button("🔄 توليد متاهة جديدة", use_container_width=True)

# إدارة الذاكرة والتوليد
if "maze" not in st.session_state or generate_btn or st.session_state.get("last_size") != grid_size or st.session_state.get("last_terrain") != enable_terrain:
    st.session_state.maze = Maze(grid_size, grid_size, add_terrain=enable_terrain)
    st.session_state.last_size = grid_size
    st.session_state.last_terrain = enable_terrain

current_maze = st.session_state.maze

# 🌟 [BONUS FEATURE 3]: أداة بناء وتعديل المتاهة يدوياً (Interactive Maze Builder)
st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ 🌟 بناء وتعديل المتاهة يدوياً (Interactive Builder)"):
    st.write("يمكنك تعديل إحداثيات خلية معينة يدوياً:")
    edit_r = st.number_input("سطر الخلية (Row):", min_value=0, max_value=grid_size-1, value=1)
    edit_c = st.number_input("عمود الخلية (Column):", min_value=0, max_value=grid_size-1, value=1)
    cell_type = st.selectbox("نوع الخلية:", ["ممر عادي (0)", "جدار (1)", "رمل - تكلفة 3 (3)", "طين - تكلفة 5 (5)"])
    
    if st.button("تطبيق التعديل على المتاهة"):
        val_map = {"ممر عادي (0)": 0, "جدار (1)": 1, "رمل - تكلفة 3 (3)": 3, "طين - تكلفة 5 (5)": 5}
        current_maze.grid[edit_r][edit_c] = val_map[cell_type]
        st.success(f"تم تعديل الخلية ({edit_r}, {edit_c}) بنجاح!")

# تنفيذ الخوارزمية المختارة وقياس الأداء
tracemalloc.start()
start_time = time.perf_counter()

if algo_name == "BFS (Breadth-First Search)":
    sol_path, total_explored = bfs(current_maze)
elif algo_name == "DFS (Depth-First Search)":
    sol_path, total_explored = dfs(current_maze)
elif algo_name == "Greedy (Manhattan Distance)":
    sol_path, total_explored = greedy_bfs(current_maze, manhattan_distance)
elif algo_name == "Greedy (Euclidean Distance)":
    sol_path, total_explored = greedy_bfs(current_maze, euclidean_distance)
elif algo_name == "A* (Manhattan Distance)":
    sol_path, total_explored = a_star(current_maze, manhattan_distance)
elif algo_name == "A* (Euclidean Distance)":
    sol_path, total_explored = a_star(current_maze, euclidean_distance)
elif algo_name == "🌟 IDA* (Iterative Deepening A* - Manhattan)":
    sol_path, total_explored = ida_star(current_maze, manhattan_distance)

end_time = time.perf_counter()
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

exec_time = (end_time - start_time) * 1000
mem_kb = peak_mem / 1024

# حساب التكلفة الكلية للمسار بناءً على التضاريس
total_path_cost = 0
if sol_path:
    for node in sol_path:
        total_path_cost += current_maze.get_cost(node)

# عرض الرسم المرئي
with col2:
    st.subheader(f"📊 العرض المرئي للمتاهة: {algo_name}")
    
    animate = st.checkbox("🔮 تفعيل محاكاة الاستكشاف خطوة بخطوة")
    
    if animate and len(total_explored) > 0:
        step = max(1, len(total_explored) // 10)
        plot_placeholder = st.empty()
        for i in range(0, len(total_explored), step):
            fig = draw_maze(current_maze, total_explored[:i])
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.03)
            
    fig = draw_maze(current_maze, total_explored, sol_path)
    st.plotly_chart(fig, use_container_width=True)

# عرض لوحة النتائج الإحصائية
st.markdown("---")
st.subheader("📈 لوحة قياس الأداء والتحليل الشامل (Performance Metrics & Terrain Cost)")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("عدد الخطوات (Steps)", len(sol_path) if sol_path else "لا يوجد")
m2.metric("التكلفة الكلية للمسار (Total Cost)", total_path_cost if sol_path else "N/A")
m3.metric("العقد المستكشفة (Nodes)", len(total_explored))
m4.metric("زمن التنفيذ (Time)", f"{exec_time:.2f} ms")
m5.metric("استهلاك الذاكرة (Memory)", f"{mem_kb:.2f} KB")

st.info("💡 دليل الألوان للتضاريس والبناء: الأبيض = ممر عادي (1) | 🟨 الأصفر = رمل (3) | 🤎 البني = طين (5) | ⬛ الأسود = جدار | 🟩 الأخضر = البداية والنهاية | 🟦 الأزرق = مساحة البحث | 🟧 البرتقالي = المسار الأمثل.")
