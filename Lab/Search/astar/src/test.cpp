#include <iostream>
#include <map>
#include <vector>
#include <set >

// 定义上下左右四个方向的偏移量
const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};

// 深度优先搜索提取相邻聚类
void dfs(int row, int col, int n, std::vector<std::vector<bool>> &matrix,
         std::vector<std::vector<bool>> &visited, int &clusterSize) {
  if (row < 0 || row >= n || col < 0 || col >= n) {
    return;
  }
  if (visited[row][col] || matrix[row][col] == 0) {
    return;
  }

  visited[row][col] = true;
  clusterSize++;

  // 检查上下左右四个方向
  for (int i = 0; i < 4; i++) {
    int newRow = row + dx[i];
    int newCol = col + dy[i];
    dfs(newRow, newCol, n, matrix, visited, clusterSize);
  }
}

// 统计每种聚类的数目
std::map<int, int> countClusters(std::vector<std::vector<bool>> &matrix) {
  int n = matrix.size();
  std::vector<std::vector<bool>> visited(
      n, std::vector<bool>(n, false)); // 用于标记已经访问过的位置
  std::map<int, int> clusterCounts;    // 统计每种聚类的数目

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (!visited[i][j] && matrix[i][j] == 1) {
        int clusterSize = 0;
        dfs(i, j, n, matrix, visited, clusterSize);
        clusterCounts[clusterSize]++; // 统计聚类的数目
      }
    }
  }

  return clusterCounts;
}

int main() {
  // 示例矩阵
  using namespace std;
  int a = 5;
  cout<<float(a)/2<<endl;
}
