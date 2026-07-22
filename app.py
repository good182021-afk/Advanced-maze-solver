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

# 1. إعدادات الصفحة التفاعلية (Full Width)
st.set_page_config(page_title="Advanced Maze Solver AI Pro", layout="wide")

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
        """إضافة تضاريس واقعية: رمل بتكلفة 3 وطين بتكلفة 5"""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 0 and (r, c) != self.start and (r, c) != self.end:
                    rand_val = random.random()
                    if rand_val < 0.15:
                        self.grid[r][c] = 3 # رمل (تكلفة 3)
                    elif rand_val < 0.25:
                        self.grid[r][c] = 5 # طين (تكلفة 5)

    def get_cost(self, pos):
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
# 3. الخوارزميات البرمجية (BFS, DFS, Greedy, A*, IDA*)
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
    for _ in range(100):
        t, found = search(path, 0, bound)
        if found:
            return path, explored
        if t == float('inf'):
            return None, explored
        bound = t
    return None, explored

# ==========================================
# 4. دالة الرسم الاحترافي للمتاهة بالتصميم الواقعي للرمل والطين
# ==========================================
def draw_maze(maze, explored, path=None):
    display_grid = np.copy(maze.grid).astype(float)
    
    # تحويل القيم إلى تمثيل ألوان واقعية
    for r in range(maze.rows):
        for c in range(maze.cols):
            val = display_grid[r, c]
            if val == 3:       # رمل
                display_grid[r, c] = 0.3
            elif val == 5:     # طين
                display_grid[r, c] = 0.5
            elif val == 0:     # ممر عادي
                display_grid[r, c] = 0.0
                
    if explored:
        for r, c in explored:
            if (r, c) != maze.start and (r, c) != maze.end:
                display_grid[r, c] = 0.65 # أزرق شفاف لمساحة البحث
            
    if path:
        for r, c in path:
            if (r, c) != maze.start and (r, c) != maze.end:
                display_grid[r, c] = 0.85 # برتقالي مشع لمسار الحل
            
    display_grid[maze.start[0], maze.start[1]] = 0.15 # بداية
    display_grid[maze.end[0], maze.end[1]] = 1.0     # نهاية

    fig = go.Figure(data=go.Heatmap(
        z=display_grid,
        colorscale=[
            [0.0, '#E2E8F0'],    # ممر عادي (أسفلت/طريق رمادي فاتح)
            [0.15, '#22C55E'],   # نقطة البداية (أخضر نيون)
            [0.3, '#FACC15'],    # 🏜️ رمل صحراوي (أصفر ذهبي)
            [0.5, '#78350F'],    # 🪵 طين ووحل (بني داكن واقعي)
            [0.65, '#3B82F6'],   # مساحة البحث واكتشاف الخوارزمية (أزرق)
            [0.85, '#F97316'],   # مسار الحل الأمثل (برتقالي مشع)
            [1.0, '#0F172A']     # الجدران والصخور (رمادي غامق/أسود)
        ],
        showscale=False,
        xgap=1.5, ygap=1.5
    ))

    # إضافة رموز وأيقونات واقعية فوق الخلايا (في الأحجام الصغرى والمتوسطة)
    if maze.rows <= 31:
        annotations = []
        for r in range(maze.rows):
            for c in range(maze.cols):
                cell_val = maze.grid[r][c]
                pos = (r, c)
                
                if pos == maze.start:
                    annotations.append(dict(x=c, y=r, text="🚀", showarrow=False, font=dict(size=14)))
                elif pos == maze.end:
                    annotations.append(dict(x=c, y=r, text="🏁", showarrow=False, font=dict(size=14)))
                elif cell_val == 3 and (not path or pos not in path):
                    annotations.append(dict(x=c, y=r, text="🏜️", showarrow=False, font=dict(size=10)))
                elif cell_val == 5 and (not path or pos not in path):
                    annotations.append(dict(x=c, y=r, text="🪵", showarrow=False, font=dict(size=10)))

        fig.update_layout(annotations=annotations)

    fig.update_layout(
        yaxis=dict(autorange="reversed", showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=620,
        paper_bgcolor='#0f172a', plot_bgcolor='#0f172a'
    )
    return fig

# ==========================================
# 5. واجهة المستخدم والتنسيق الهيكلي المطلوب
# ==========================================

# 1️⃣ العنوان العلوي الرئيسي
st.markdown("<h1 style='text-align: center;'>🧠 Advanced AI Maze Solver Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>نظام حل المتاهات الذكي والمقارنة بين خوارزميات البحث الملاحة</p>", unsafe_allow_html=True)
st.markdown("---")

# 2️⃣ إعدادات المحاكاة في أعلى الصفحة
st.subheader("⚙️ إعدادات المحاكاة وتوليد المتاهة")
c1, c2, c3, c4 = st.columns([1.5, 2, 1.5, 1])

with c1:
    size_option = st.selectbox("📏 حجم المتاهة:", ["11x11 (صغير)", "31x31 (متوسط)", "51x51 (كبير)"])
    size_dict = {"11x11 (صغير)": 11, "31x31 (متوسط)": 31, "51x51 (كبير)": 51}
    grid_size = size_dict[size_option]

with c2:
    algo_name = st.selectbox("🧠 خوارزمية البحث:", [
        "BFS (Breadth-First Search)", 
        "DFS (Depth-First Search)", 
        "Greedy (Manhattan Distance)", 
        "Greedy (Euclidean Distance)",
        "A* (Manhattan Distance)",
        "A* (Euclidean Distance)",
        "🌟 IDA* (Iterative Deepening A* - Manhattan)"
    ])

with c3:
    enable_terrain = st.checkbox("🏜️ تفعيل التضاريس الواقعية (رمل/طين)", value=False)
    animate = st.checkbox("🔮 محاكاة الاستكشاف (أنيميشن)", value=False)

with c4:
    st.write("")
    generate_btn = st.button("🔄 توليد جديد", use_container_width=True)

# أداة بناء وتعديل المتاهة يدوياً
with st.expander("🛠️ 🌟 بناء وتعديل المتاهة يدوياً (Interactive Builder)"):
    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1: edit_r = st.number_input("سطر الخلية (Row):", min_value=0, max_value=grid_size-1, value=1)
    with ec2: edit_c = st.number_input("عمود الخلية (Column):", min_value=0, max_value=grid_size-1, value=1)
    with ec3: cell_type = st.selectbox("نوع الخلية:", ["ممر عادي (0)", "جدار (1)", "رمل - تكلفة 3 (3)", "طين - تكلفة 5 (5)"])
    with ec4: 
        st.write("")
        apply_edit = st.button("تطبيق التعديل")

# إدارة حالة المتاهة
if "maze" not in st.session_state or generate_btn or st.session_state.get("last_size") != grid_size or st.session_state.get("last_terrain") != enable_terrain:
    st.session_state.maze = Maze(grid_size, grid_size, add_terrain=enable_terrain)
    st.session_state.last_size = grid_size
    st.session_state.last_terrain = enable_terrain

current_maze = st.session_state.maze

if 'apply_edit' in locals() and apply_edit:
    val_map = {"ممر عادي (0)": 0, "جدار (1)": 1, "رمل - تكلفة 3 (3)": 3, "طين - تكلفة 5 (5)": 5}
    current_maze.grid[edit_r][edit_c] = val_map[cell_type]
    st.success(f"تم تعديل الخلية ({edit_r}, {edit_c}) بنجاح!")

# حساب وتنفيذ الخوارزمية المحددة
tracemalloc.start()
start_time = time.perf_counter()

if algo_name == "BFS (Breadth-First Search)": sol_path, total_explored = bfs(current_maze)
elif algo_name == "DFS (Depth-First Search)": sol_path, total_explored = dfs(current_maze)
elif algo_name == "Greedy (Manhattan Distance)": sol_path, total_explored = greedy_bfs(current_maze, manhattan_distance)
elif algo_name == "Greedy (Euclidean Distance)": sol_path, total_explored = greedy_bfs(current_maze, euclidean_distance)
elif algo_name == "A* (Manhattan Distance)": sol_path, total_explored = a_star(current_maze, manhattan_distance)
elif algo_name == "A* (Euclidean Distance)": sol_path, total_explored = a_star(current_maze, euclidean_distance)
elif algo_name == "🌟 IDA* (Iterative Deepening A* - Manhattan)": sol_path, total_explored = ida_star(current_maze, manhattan_distance)

end_time = time.perf_counter()
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

exec_time = (end_time - start_time) * 1000
mem_kb = peak_mem / 1024

total_path_cost = sum(current_maze.get_cost(node) for node in sol_path) if sol_path else 0

st.markdown("---")

# 3️⃣ الرسم في منتصف الصفحة تماماً
st.markdown("<h3 style='text-align: center;'>🗺️ العرض المرئي للمتاهة ورسم المسار</h3>", unsafe_allow_html=True)

mid_col1, mid_col2, mid_col3 = st.columns([1, 4, 1])

with mid_col2:
    if animate and len(total_explored) > 0:
        step = max(1, len(total_explored) // 10)
        plot_placeholder = st.empty()
        for i in range(0, len(total_explored), step):
            fig = draw_maze(current_maze, total_explored[:i])
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.03)
            
    fig = draw_maze(current_maze, total_explored, sol_path)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 4️⃣ قسم التحليل والنتائج أسفل الرسم في نهاية الصفحة
st.subheader("📈 لوحة قياس الأداء والتحليل الإحصائي (Performance Metrics)")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("طول المسار (Steps)", len(sol_path) if sol_path else "لا يوجد")
m2.metric("التكلفة الكلية (Total Cost)", total_path_cost if sol_path else "N/A")
m3.metric("العقد المستكشفة (Nodes)", len(total_explored))
m4.metric("زمن التنفيذ (Time)", f"{exec_time:.2f} ms")
m5.metric("استهلاك الذاكرة (Memory)", f"{mem_kb:.2f} KB")

st.caption("💡 دليل التضاريس الواقعية: الطريق الرمادي = ممر عادي (1) | 🏜️ الأصفر = رمل (3) | 🪵 البني = طين (5) | ⬛ الداكن = جدار | 🚀/🏁 الأخضر = البداية والنهاية | 🟦 الأزرق = نطاق البحث | 🟧 البرتقالي = المسار الأفضل.")

st.markdown("<br><br><br>", unsafe_allow_html=True)

# 5️⃣ التوقيع الأكاديمي أسفل الصفحة على اليسار
footer_col1, footer_col2 = st.columns([3, 1])
with footer_col2:
    st.markdown("""
    <div style='text-align: left; color: #94a3b8; font-size: 0.85rem; border-left: 2px solid #3b82f6; padding-left: 10px;'>
        <b>مشروع مادة الذكاء الاصطناعي</b><br>
        <b>إعداد الطالبات:</b> هبة محمود & فاطمة خليفة<br>
        <b>تحت إشراف:</b> د. صلاح النعاس
    </div>
    """, unsafe_allow_html=True)
