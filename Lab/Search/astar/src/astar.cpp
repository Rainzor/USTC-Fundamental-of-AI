#include "astar.hpp"
#define FILENUM 10
#define INPUTNAME "../input/input.txt"
#define OUTPUTNAME "../output/output.txt"
using namespace std;

int getInput(vector<vector<bool>> &input, const string filename);
void getOutput(vector<vector<bool>> &input, int n, const string filename);
int main() {
  string ifilename = INPUTNAME;
    string ofilename = OUTPUTNAME;
    int file_n = 0;
    int split_input_pos = ifilename.find(".txt");
    int split_output_pos = ofilename.find(".txt");
    while (file_n < FILENUM) {
        ifilename = INPUTNAME;
        ifilename.insert(split_input_pos, std::to_string(file_n));
        vector<vector<bool>> input;
        int n = getInput(input, ifilename);
        // std::cout << ifilename << std::endl;
        // for(int i=0;i<n;i++){
        //     for(int j=0;j<n;j++){
        //         std::cout<<input[i][j]<<" ";
        //     }
        //     std::cout<<std::endl;
        // }
        ofilename = OUTPUTNAME;
        ofilename.insert(split_output_pos, std::to_string(file_n));

        getOutput(input, n, ofilename);
        file_n++;
    }
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

void getOutput(vector<vector<bool>> &input, int n, const string filename) {
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //     std::cout << input[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    ofstream file;
    vector<Operation> op_list;
    vector<Operation> operations;
    State start_state;
    Board start_board;
    // vector<int> input_temp(n * n);
    // for (int i = 0; i < n * n; i++) {
    //   input_temp[i] = input[i / n][i % n]?1:0;
    // }
    file.open(filename);
    if (!file.is_open()) {
        cout << "Error opening file" << endl;
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
        if (input[i][j]) {
            start_state.insert(Position(i, j));
        }
        }
    }
    start_board = Board(start_state, n);
    if (n <= 6) {
        operations = astar(start_board);
        // operations = IDAstar(start_board);

        for (auto &op : operations) {
        op_list.push_back(op);
        }
    } else {
        int sub_width = 6;
        int sub_num = ceil(float(n) / sub_width);
        int target = min(int(start_state.size() / 2), n * n / 3);
        vector<Operation> operations = IDAstar(start_board, target);
        for (int i = 0; i < operations.size(); i++) {
            op_list.push_back(operations[i]);
        }
        State &state_temp = start_board.state;
        for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (state_temp.find(Position(i, j)) == state_temp.end()) {
                input[i][j] = false;
            }
        }
        }

        for (int i = 0; i < sub_num; i++) {
        for (int j = 0; j < sub_num; j++) {
            Position pos(i * sub_width, j * sub_width);
            start_state.clear();
            if (pos.first + sub_width >= n)
                pos.first = n - sub_width;
            if (pos.second + sub_width >= n)
                pos.second = n - sub_width;
            for (int k = 0; k < sub_width; k++) {
                for (int l = 0; l < sub_width; l++) {
                    if (input[(pos.first + k)][pos.second + l]) {
                        start_state.insert(Position(k, l));
                        input[(pos.first + k)][pos.second + l] = false;
                    }
                }
            }
            start_board = Board(start_state, sub_width);
            //operations = astar(start_board);
            operations = IDAstar(start_board);
            for (auto &op : operations) {
                op.first.first += pos.first;
                op.first.second += pos.second;
                op_list.push_back(op);
            }
        }
        }
    }
    if (!op_list.empty()) {
        file << op_list.size() << endl;
        for (auto &op : op_list) {
        file << op.first.first << ", " << op.first.second << ", ";
        switch (op.second) {
        case Direction::UP:
            file << 1 << endl;
            break;
        case Direction::LEFT:
            file << 2 << endl;
            break;
        case Direction::DOWN:
            file << 3 << endl;
            break;
        case Direction::RIGHT:
            file << 4 << endl;
            break;
        }
        }
    } else {
        file << "No solution!" << endl;
    }

    file.close();
}
