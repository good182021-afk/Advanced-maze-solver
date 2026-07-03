import streamlit as st
import numpy as np
import random
import time
import tracemalloc
import math
from collections import deque
import heapq
import plotly.graph_objects as go

# إعدادات الصفحة التفاعلية
st.set_page_config(page_title="Advanced Maze Solver AI", layout="wide")
st.title(" 🧠 Advanced Maze Solver Using AI Search Algorithms")
st.markdown("### مشروع مادة الذكاء الاصطناعي - إعداد: هبة محمود & فاطمة خليفة | تحت إشراف: د. صلاح")

# ==========================================
# 1. كلاس توليد المتاهة
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
# 3. خوارزميات البحث مع تسجيل خطوات الاستكشاف للرسم التوضيحي
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
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic_func(neighbor, end)
                came_from[neighbor] = current
                heapq.heappush(open_set, (f_score, neighbor))
    return None, explored

# ==========================================
# 4. دالة الرسم الاحترافي للمتاهة والمسار المتحرك
# ==========================================
def draw_maze(maze, explored, path=None):
    # إنشاء مصفوفة ألوان للعرض بصرياً
    # 0: ممر (أبيض), 1: جدار (أسود), 2: استكشاف (أزرق خفيف), 3: مسار الحل (أخضر)
    display_grid = np.copy(maze.grid).astype(float)
    
    if explored:
        for r, c in explored:
            display_grid[r, c] = 0.5 # تم استكشافه
            
    if path:
        for r, c in path:
            display_grid[r, c] = 0.8 # مسار الحل الفعلي
            
    display_grid[maze.start[0], maze.start[1]] = 0.2 # البداية
    display_grid[maze.end[0], maze.end[1]] = 1.0 # النهاية

    fig = go.Figure(data=go.Heatmap(
        z=display_grid,
        colorscale=[
            [0.0, 'white'],      # الممرات
            [0.2, '#10b981'],    # البداية (أخضر زاهي)
            [0.5, '#60a5fa'],    # العقد المستكشفة (أزرق سماوي)
            [0.8, '#f59e0b'],    # مسار الحل (برتقالي ذهبي)
            [1.0, '#1e293b']     # الجدران (رمادي غامق جداً)
        ],
        showscale=False,
        xgap=1, ygap=1
    ))
    
    fig.update_layout(
        yaxis=dict(autorange="reversed", showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=600,
        width=600
    )
    return fig

# ==========================================
# 5. واجهة المستخدم الرسومية في المتصفح
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.sidebar.header("⚙️ إعدادات المحاكاة")
    size_option = st.sidebar.selectbox("اختر حجم المصفوفة (المتاهة):", ["11x11 (صغير)", "51x51 (متوسط)", "101x101 (كبير)"])
    size_dict = {"11x11 (صغير)": 11, "51x51 (متوسط)": 51, "101x101 (كبير)": 101}
    grid_size = size_dict[size_option]
    
    algo_name = st.sidebar.selectbox("اختر خوارزمية البحث:", [
        "BFS (Breadth-First Search)", 
        "DFS (Depth-First Search)", 
        "Greedy (Manhattan Distance)", 
        "Greedy (Euclidean Distance)",
        "A* (Manhattan Distance)",
        "A* (Euclidean Distance)"
    ])
    
    generate_btn = st.sidebar.button("🔄 توليد متاهة عشوائية جديدة")

# حفظ المتاهة في الذاكرة عشان ما تتغيرش مع كل ضغطة زر
if "maze" not in st.session_state or generate_btn or st.session_state.get("last_size") != grid_size:
    st.session_state.maze = Maze(grid_size, grid_size)
    st.session_state.last_size = grid_size

current_maze = st.session_state.maze

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

end_time = time.perf_counter()
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

exec_time = (end_time - start_time) * 1000
mem_kb = peak_mem / 1024

# عرض الرسم المرئي
with col2:
    st.subheader(f"📊 العرض المرئي للمتاهة باستخدام خوارزمية: {algo_name}")
    
    # دالة اختيارية لمحاكاة الحركة البصرية (الأنيميشن)
    animate = st.checkbox("🔮 تفعيل محاكاة الاستكشاف خطوة بخطوة (بطيء في الأحجام الكبيرة)")
    
    if animate and len(total_explored) > 0:
        # رسم عينة متحركة سريعة لتبسيط العرض والسرعة
        step = max(1, len(total_explored) // 10)
        plot_placeholder = st.empty()
        for i in range(0, len(total_explored), step):
            fig = draw_maze(current_maze, total_explored[:i])
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.05)
            
    # عرض الحل النهائي الثابت والجميل
    fig = draw_maze(current_maze, total_explored, sol_path)
    st.plotly_chart(fig, use_container_width=True)

# عرض لوحة النتائج الإحصائية ومقارنة الأداء تحت الرسم
st.markdown("---")
st.subheader("📈 لوحة قياس الأداء والتحليل (Performance Metrics)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("طول المسار الناتج (Path Length)", len(sol_path) if sol_path else "لا يوجد")
m2.metric("عدد العقد المستكشفة (Nodes Explored)", len(total_explored))
m3.metric("زمن التنفيذ (Execution Time)", f"{exec_time:.2f} ms")
m4.metric("استهلاك الذاكرة (Memory Usage)", f"{mem_kb:.2f} KB")

st.info("💡 دليل الألوان: الأبيض ممرات | الأسود جدران | الأخضر نقطة البداية والنهاية | الأزرق يوضح المساحة التي بحثت فيها الخوارزمية | البرتقالي يمثل المسار الأقصر الذي تم اختياره.")