#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <vector>

using namespace std;

enum class Direction { UP, LEFT, DOWN, RIGHT };

typedef std::pair<int, int> Position;
typedef std::set<Position> State;
typedef std::pair<Position, Direction> Operation;

// 定义上下左右四个方向的偏移量
const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};

class Board {
public:
    State state;
    Board *parent = nullptr;
    int width;
    int g = 0;
    int h = 0;
    Operation operation;
    std::map<int, int> clusterCounts;

public:
    Board() {
        g = 0;
        h = 0;
        parent = nullptr;
        width = 0;
    }
    Board(State cur_state, Board *prt, int w, int g, int h, Operation op)
        : state(cur_state), parent(prt), width(w), g(g), h(h), operation(op) {
        getClusters();
    }
    Board(State cur_state, int w, bool flag = true) : state(cur_state), width(w) {
        parent = nullptr;
        if (flag) {
        getClusters();
        h = getH(); // 启发式函数
        }
        g = 0;
    }
    Board(State cur_state, Board *prt, int w, Operation op, bool flag = true)
        : state(cur_state), parent(prt), width(w), operation(op) {
        if (flag) {
        getClusters();
        h = getH();
        }
        g = getG();
    }
    Board(const Board &other) {
        state = other.state;
        parent = other.parent;
        width = other.width;
        g = other.g;
        h = other.h;
        operation = other.operation;
        clusterCounts = other.clusterCounts;
    }
    virtual ~Board() { parent = nullptr; }
    void setParent(Board *prt) {
        parent = prt;
        g = getG();
    }

    int getG() {

        if (parent == nullptr)
        return 0;
        else {
        return parent->g + 1;

        // size_t max_clus = 0;
        // for (auto &c : clusterCounts) {
        //     if(c.first>max_clus)
        // max_clus = c.first;
        // }
        // if (max_clus > 3 && width > 3)
        //   return parent->g + 3;
        // else
        //   return parent->g + 1;
        }
    }

    int getH() {
        int H = 0;
        float s = float(state.size());
        //H = int(s / 3);
        if ((width > 4 && (s > int(width * width / 2))))
        H = int(s / 2);
        else {

        for (auto &p : clusterCounts) {
            if (p.first > 6)
            H += p.second * p.first / 3;
            else {
            switch (p.first) {
            case 1:
                H += p.second + (p.second % 2) * 2;
                break;
            case 2:
                H += p.second + (p.second % 2) * 1;
                break;
            case 4:
                H += p.second * 2;
                break;
            case 5:
                H += p.second * 2;
                break;
            default:
                H += p.second * (p.first / 3);
                break;
            }
            // H += p.second;
            }
        }
        }
        return H;
    }
    int getF() { return g + h; }
    State getStateChanged(Operation op) {
        Position pos = op.first;
        Direction direc = op.second;
        State new_state = state;

        vector<Position> vec_pos;
        switch (direc) {
        case Direction::UP:
        vec_pos = vector<Position>{Position(pos.first, pos.second),
                                    Position(pos.first, pos.second + 1),
                                    Position(pos.first - 1, pos.second)};
        break;
        case Direction::LEFT:
        vec_pos = vector<Position>{Position(pos.first, pos.second),
                                    Position(pos.first - 1, pos.second),
                                    Position(pos.first, pos.second - 1)};
        break;
        case Direction::DOWN:
        vec_pos = vector<Position>{Position(pos.first, pos.second),
                                    Position(pos.first, pos.second - 1),
                                    Position(pos.first + 1, pos.second)};
        break;
        case Direction::RIGHT:
        vec_pos = vector<Position>{Position(pos.first, pos.second),
                                    Position(pos.first + 1, pos.second),
                                    Position(pos.first, pos.second + 1)};
        break;
        }
        for (auto &p : vec_pos) {
        if (new_state.find(p) != new_state.end()) {
            new_state.erase(p);
        } else {
            new_state.insert(Position(p.first, p.second));
        }
        }
        return new_state;
    }

    bool operator==(const Board &other) { return state == other.state; }

    void draw() {
        for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            if (state.find(Position(i, j)) != state.end()) {
            cout << "1 ";
            } else {
            cout << "0 ";
            }
        }
        cout << endl;
        }
    }

    void getCandidateOperation(vector<Operation> &op_list) {
        op_list.clear();
        // op_list.reserve(state.size() * 12);
        for (auto &pos : state) {
        if (pos.first > 0 && pos.second + 1 < width) {
            op_list.push_back(Operation(pos, Direction::UP));
            op_list.push_back(
                Operation(Position(pos.first - 1, pos.second), Direction::RIGHT));
            op_list.push_back(
                Operation(Position(pos.first, pos.second + 1), Direction::LEFT));
        }
        if (pos.first > 0 && pos.second > 0) {
            op_list.push_back(Operation(pos, Direction::LEFT));
            op_list.push_back(
                Operation(Position(pos.first - 1, pos.second), Direction::DOWN));
            op_list.push_back(
                Operation(Position(pos.first, pos.second - 1), Direction::UP));
        }
        if (pos.first + 1 < width && pos.second > 0) {
            op_list.push_back(Operation(pos, Direction::DOWN));
            op_list.push_back(
                Operation(Position(pos.first + 1, pos.second), Direction::LEFT));
            op_list.push_back(
                Operation(Position(pos.first, pos.second - 1), Direction::RIGHT));
        }
        if (pos.first + 1 < width && pos.second + 1 < width) {
            op_list.push_back(Operation(pos, Direction::RIGHT));
            op_list.push_back(
                Operation(Position(pos.first + 1, pos.second), Direction::UP));
            op_list.push_back(
                Operation(Position(pos.first, pos.second + 1), Direction::DOWN));
        }
        }
        sort(op_list.begin(), op_list.end());
        auto uniqueIter = unique(op_list.begin(), op_list.end());
        op_list.erase(uniqueIter, op_list.end());
    }

    private:
    // 深度优先搜索提取相邻聚类
    void dfs_find_cluster(int row, int col, int n,
                            const std::vector<std::vector<bool>> &matrix,
                            std::vector<std::vector<bool>> &visited,
                            int &clusterSize) {
        if (row < 0 || row >= n || col < 0 || col >= n) {
        return;
        }
        if (visited[row][col] || matrix[row][col] == false) {
        return;
        }

        visited[row][col] = true;
        clusterSize++;

        // 检查上下左右四个方向
        for (int i = 0; i < 4; i++) {
        int newRow = row + dx[i];
        int newCol = col + dy[i];
        dfs_find_cluster(newRow, newCol, n, matrix, visited, clusterSize);
        }
    }

    // 查找聚类并统计每种聚类的数目
    void getClusters() {
        int n = width;
        std::vector<std::vector<bool>> matrix(
            n, std::vector<bool>(n, false)); // 用于标记已经访问过的位置

        for (auto &pos : state) {
        matrix[pos.first][pos.second] = true;
        }

        std::vector<std::vector<bool>> visited(
            n, std::vector<bool>(n, false)); // 用于标记已经访问过的位置
        for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (!visited[i][j] && matrix[i][j] == true) {
            int clusterSize = 0;
            dfs_find_cluster(i, j, n, matrix, visited, clusterSize);
            clusterCounts[clusterSize]++; // 统计聚类的数目
            }
        }
        }
    }
};

int isStateInList(vector<Board *> &blist, State &state);

vector<Operation> astar(const Board &start);

vector<Operation> dijkstraSearch(const Board &start);

int IDAsearch(vector<Board *> &path, set<State> &open_set, int g, int bound,
              int &loop, const int &target);
vector<Operation> IDAstar(Board &start, int target = 0);

int isStateInList(vector<Board *> &blist, State &state) {
    int i = 0;
    for (auto &b : blist) {
        if (b->state == state) {
        return i;
        }
        i++;
    }
    return -1;
}

vector<Operation> astar(const Board &start) {
    vector<Board *> open_list;  // 用于存储待访问的Board指针
    vector<Board *> close_list; // 用于存储已经访问过的Board指针
    vector<Operation> operation_list;
    vector<Operation> path_op_list;
    set<State> open_set;  // 用于存储待访问的状态State
    set<State> close_set; // 用于存储已经访问过的状态State
    int loop = 0;
    Board *start_board = new Board(start);
    open_list.push_back(start_board);
    open_set.insert(start_board->state);
    int max_depth = int(start.state.size() / 2);
    // max_depth = max_depth > 10 ? max_depth + 3 : 20;
    max_depth = INT32_MAX;

    while (!open_list.empty()) {
        loop++;
        Board *cur_board = open_list[0];
        // 如果当前状态为空,即所有的格子都被消除了,抵达目标状态
        if (cur_board->state.size() == 0) {
        cout << "A* loop: " << loop << endl;
        // return *cur_board;
        Board *p = cur_board;
        // p->draw();
        // cout<<endl;
        while (p->parent != nullptr) {
            path_op_list.push_back(p->operation);
            p = p->parent;
            // p->draw();
            // cout<<endl;
        }
        reverse(path_op_list.begin(), path_op_list.end());
        for (auto &b : open_list) {
            if (b != nullptr)
            delete b;
        }
        for (auto &b : close_list) {
            if (b != nullptr)
            delete b;
        }
        return path_op_list;
        }
        if (cur_board->g < max_depth) {
        cur_board->getCandidateOperation(operation_list);
        for (auto &op : operation_list) {
            State new_state = cur_board->getStateChanged(op);

            bool in_open_set = open_set.find(new_state) != open_set.end();
            bool in_close_set = close_set.find(new_state) != close_set.end();

            if (!in_open_set &&
                !in_close_set) { // 如果不在close_list中,也不在open_list中
            Board *new_board =
                new Board(new_state, cur_board, cur_board->width, op);
            open_list.push_back(new_board);
            open_set.insert(new_state);
            } else if (in_open_set) {
            Board *old_board = open_list[isStateInList(open_list, new_state)];
            if (old_board->g > cur_board->g + 1) {
                old_board->setParent(cur_board);
                old_board->operation = op;
            }
            } else if (in_close_set) {
            int idx = isStateInList(close_list, new_state);
            Board *old_board = close_list[idx];
            if (old_board->g > cur_board->g + 1) {
                old_board->setParent(cur_board);
                old_board->operation = op;
                open_list.push_back(old_board);
                open_set.insert(new_state);
                close_list.erase(close_list.begin() + idx);
                close_set.erase(new_state);
            }
            }
        }
        }
        open_list.erase(open_list.begin());
        open_set.erase(cur_board->state);
        close_list.push_back(cur_board);
        close_set.insert(cur_board->state);
        sort(open_list.begin(), open_list.end(),
             [](Board *a, Board *b) { return a->getF() < b->getF(); });
    }
    cout << "loop: " << loop << " ,no solution" << endl;
    // output_file << "No solution" << endl;
    for (auto &b : close_list) {
        delete b;
    }
    return path_op_list;
}

vector<Operation> dijkstraSearch(const Board &start) {
    vector<Board *> open_list;  // 用于存储待访问的Board指针
    vector<Board *> close_list; // 用于存储已经访问过的Board指针
    vector<Operation> operation_list;
    vector<Operation> path_op_list;
    set<State> open_set;  // 用于存储待访问的状态State
    set<State> close_set; // 用于存储已经访问过的状态State
    int loop = 0;
    Board *start_board = new Board(start);
    open_list.push_back(start_board);
    open_set.insert(start_board->state);
    int max_depth = int(start.state.size() / 2);
    // max_depth = max_depth > 10 ? max_depth + 3 : 20;
    max_depth = INT32_MAX;

    while (!open_list.empty()) {
        loop++;
        Board *cur_board = open_list[0];
        // 如果当前状态为空,即所有的格子都被消除了,抵达目标状态
        if (cur_board->state.size() == 0) {
        cout << "A* loop: " << loop << endl;
        // return *cur_board;
        Board *p = cur_board;
        // p->draw();
        // cout<<endl;
        while (p->parent != nullptr) {
            path_op_list.push_back(p->operation);
            p = p->parent;
            // p->draw();
            // cout<<endl;
        }
        reverse(path_op_list.begin(), path_op_list.end());
        for (auto &b : open_list) {
            if (b != nullptr)
            delete b;
        }
        for (auto &b : close_list) {
            if (b != nullptr)
            delete b;
        }
        return path_op_list;
        }
        if (cur_board->g < max_depth) {
        cur_board->getCandidateOperation(operation_list);
        for (auto &op : operation_list) {
            State new_state = cur_board->getStateChanged(op);

            bool in_open_set = open_set.find(new_state) != open_set.end();
            bool in_close_set = close_set.find(new_state) != close_set.end();

            if (!in_open_set &&
                !in_close_set) { // 如果不在close_list中,也不在open_list中
            Board *new_board =
                new Board(new_state, cur_board, cur_board->width, op, false);
            open_list.push_back(new_board);
            open_set.insert(new_state);
            } else if (in_open_set) {
            Board *old_board = open_list[isStateInList(open_list, new_state)];
            if (old_board->g > cur_board->g + 1) {
                old_board->setParent(cur_board);
                old_board->operation = op;
            }
            } else if (in_close_set) {
            continue;
            }
        }
        }
        open_list.erase(open_list.begin());
        open_set.erase(cur_board->state);
        close_list.push_back(cur_board);
        close_set.insert(cur_board->state);
        sort(open_list.begin(), open_list.end(),
             [](Board *a, Board *b) { return a->g < b->g; });
    }
    cout << "loop: " << loop << " ,no solution" << endl;
    // output_file << "No solution" << endl;
    for (auto &b : close_list) {
        delete b;
    }
    return path_op_list;
}

int IDAsearch(vector<Board *> &path, set<State> &open_set, int g, int bound,
              int &loop, const int &target) {
    Board *cur_board = path.back();
    int f = g + cur_board->getH();
    if (f > bound) {
        return f;
    }
    if (cur_board->state.size() <= target) {
        return -1;
    }
    loop++;
    int min = INT32_MAX;
    vector<Operation> operation_list;
    cur_board->getCandidateOperation(operation_list);
    for (auto &op : operation_list) {
        State new_state = cur_board->getStateChanged(op);

        if (open_set.find(new_state) == open_set.end()) {
        Board *new_board =
            new Board(new_state, cur_board, cur_board->width, op);
        path.push_back(new_board);
        open_set.insert(new_state);
        int t = IDAsearch(path, open_set, g + 1, bound, loop, target);
        if (t == -1) {
            return -1;
        }
        if (t < min) {
            min = t;
        }
        Board *delete_board = path.back();
        path.pop_back();
        open_set.erase(delete_board->state);
        delete delete_board;
        }
    }
    return min;
}

vector<Operation> IDAstar(Board &start, int target) {
    int loop;
    set<State> open_set;
    vector<Board *> path;
    Board *start_board = new Board(start);
    vector<Operation> path_op_list;
    path.push_back(start_board);
    open_set.clear();
    open_set.insert(start_board->state);
    int bound = start_board->h;
    loop = 0;
    while (true) {
        int t = IDAsearch(path, open_set, 0, bound, loop, target);
        if (t == -1) {
        cout << "IDA* loop: " << loop << endl;

        for (auto &b : path) {
            path_op_list.push_back(b->operation);
        }
        start.state = path.back()->state;
        path_op_list.erase(path_op_list.begin());
        for (auto &b : path) {
            delete b;
        }
        return path_op_list;
        }
        if (t == INT32_MAX) {
        cout << "no solution" << endl;
        for (auto &b : path) {
            delete b;
        }
        return path_op_list;
        }
        bound = t;
    }
    return path_op_list;
}
