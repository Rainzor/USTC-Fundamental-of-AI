#include "astar.hpp"
#include <chrono>
#define INPUTNAME "../input/input2.txt"
using namespace std;
int getInput(vector<vector<bool>> &input, const string filename);
void compare(vector<vector<bool>> &input, int n);

int main(){
  string ifilename = INPUTNAME;
  vector<vector<bool>> input;
  int n = getInput(input, ifilename);
  compare(input, n);
  return 0;
}




int getInput(vector<vector<bool>> &input, const string filename) {
  ifstream file;
  file.open(filename);
  if (!file.is_open()) {
    cout << "Error opening file" << endl;
    exit(1);
  }
  string line;
  int n;
  file >> n;
  input.resize(n);
  for (int i = 0; i < n; i++) {
    input[i].resize(n);
  }
  getline(file, line); // 读取换行符
  for (int i = 0; i < n; i++) {
    getline(file, line);
    for (int j = 0; j < n; j++) {
      input[i][j] = line[j * 2] - '0';
    }
  }
  file.close();
  return n;
}

void compare(vector<vector<bool>> &input, int n) {
  // for (int i = 0; i < n; i++) {
  //     for (int j = 0; j < n; j++) {
  //     std::cout << input[i][j] << " ";
  //     }
  //     std::cout << std::endl;
  // }
  vector<Operation> op_list;
  vector<Operation> operations;
  State start_state;
  Board start_board;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (input[i][j]) {
        start_state.insert(Position(i, j));
      }
    }
  }
  start_board = Board(start_state, n);
  
  cout<<"Coparing A* and Dijkstra"<<endl;
  cout<<"Board size: "<<n<<"*"<<n<<endl;
  auto start = chrono::high_resolution_clock::now();
  operations = astar(start_board);
  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
  cout<<"A* time: "<<duration.count()<<" ms"<<endl;
  
  start = chrono::high_resolution_clock::now();
  operations = dijkstraSearch(start_board);
  end = chrono::high_resolution_clock::now();
  duration = chrono::duration_cast<chrono::milliseconds>(end - start);
  cout << "Dijkstra time: " << duration.count()<< " ms" << endl;

    // operations = IDAstar(start_board);
  
}