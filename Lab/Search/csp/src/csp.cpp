#include "csp.hpp"
#include <chrono>

#define FILENUM 10
#define INPUTNAME "../input/input.txt"
#define OUTPUTNAME "../output/output.txt"

using namespace std;
void getInput(vector<Worker> &, const string );
void getOutput(CSP& csp, const string filename);
int main(){
    string ifilename = INPUTNAME;
    string ofilename = OUTPUTNAME;
    int file_n = 0;
    int split_input_pos = ifilename.find(".txt");
    int split_output_pos = ofilename.find(".txt");

    while (file_n < FILENUM) {
        ifilename = INPUTNAME;
        ifilename.insert(split_input_pos, std::to_string(file_n));
        vector<Worker> workers;

        getInput(workers, ifilename);
        auto start = chrono::system_clock::now();
        //CSP WORKING
        CSP csp(workers);
        bool find = csp.backtracking();
        int meet_request = 0;
        ofilename = OUTPUTNAME;
        ofilename.insert(split_output_pos, std::to_string(file_n));
        getOutput(csp, ofilename);
        file_n++;
        auto end = chrono::system_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(end - start);
        cout<<"Time cost: "<<duration.count()<<"s"<<endl<<endl;
    }
}
void getInput(vector<Worker> &workers,
              const string filename) {
    int workers_num;
    int days;
    int shifts_per_day;
    ifstream fin(filename);
    if(!fin){
        cout<<"File not found"<<endl;
        exit(1);
    }
    string line;
    string token;
    vector<int> tmp;
    getline(fin, line);
    std::stringstream ss(line);
    while (getline(ss, token, ',')) {
        tmp.push_back(stoi(token));
    }
    workers_num = tmp[0];
    days = tmp[1];
    shifts_per_day = tmp[2];
    cout<<"file: "<<filename<<endl;
    cout << "workers_num: " << workers_num;
    cout << ", days: " << days;
    cout << ", shifts_per_day: " << shifts_per_day << endl;

    workers.resize(workers_num);
    for (int i = 0; i < workers_num; i++) {
        workers[i].days.resize(days);
        for(int j=0;j<days;j++){
            workers[i].days[j].reserve(shifts_per_day);
            getline(fin, line);
            std::stringstream ss(line);
            while (getline(ss, token, ',')) {
                workers[i].days[j].push_back(stoi(token));
            }
        }
    }

//    for(int i=0;i<workers_num;i++){
//        for(int j=0;j<days;j++){
//            for(int k=0;k<shifts_per_day;k++){
//                cout<<workers[i].days[j][k]<<" ";
//            }
//            cout<<endl;
//        }
//        cout<<endl;
//    }

    fin.close();

}

void getOutput(CSP& csp, const string filename){
    ofstream fout(filename);
    if(!fout){
        cout<<"File not found"<<endl;
        exit(1);
    }
    bool find = csp.backtracking();
    int meet_request = 0;
    if(find) {
        meet_request = csp.printShifts(fout);
        fout << meet_request << endl;
    }else{
        fout<<"No valid schedule found"<<endl;
    }
    fout.close();
    if(find){
        printf("Solution found!\n");
        printf("Total requests: %d\t",csp.size);
        printf("Meet requests: %d\n",meet_request);
    }else {
        printf("No valid schedule found\n");
    }
}


int CSP::backtracking() {
    int shift_id = -1;
    vector<int> unassigned;
    bool isAssign =  selectUnassignedMRV(shift_id,unassigned);
    if (!isAssign) {//isAssign==false说明此次分配失败
        return 0;
    }
    if (shift_id == -1) {//shift_id==-1说明所有班次都被分配了
        return 1;
    }
    vector<set<int>> candidate_sets_backup(candidate_sets);
    int free_shifts_num_backup = free_shifts_num;
    vector<int> open_list;
    // 优先考虑交集
    if(intersection_sets[shift_id].size()==0){
        for(auto& candidate:candidate_sets[shift_id]){
            open_list.push_back(candidate);
        }
    }else{
        for(auto& candidate:intersection_sets[shift_id]){
            open_list.push_back(candidate);
        }
        for(auto& candidate:candidate_sets[shift_id]){
            if(intersection_sets[shift_id].find(candidate)==intersection_sets[shift_id].end()){
                open_list.push_back(candidate);
            }
        }
    }
    for(int i=0;i<unassigned.size();i++){
        if(unassigned[i]==shift_id){
            unassigned.erase(unassigned.begin()+i);
            break;
        }
    }

    for (auto &worker_id : open_list) {
        shifts_for_workers[shift_id] = worker_id;
        workers_shift_num[worker_id]++;
        bool check = forwardChecking(shift_id, worker_id, unassigned);
        //bool check=true;
        if (check&&backtracking() == 1) {
            return 1;
        }
        shifts_for_workers[shift_id] = -1;
        workers_shift_num[worker_id]--;
        candidate_sets = candidate_sets_backup;
        free_shifts_num = free_shifts_num_backup;
    }
    return 0;
}


bool CSP::forwardChecking(const int &shift_id, const int &worker_id,const vector<int> &unassigned) {
    // 检查约束
    // 1:相邻班次不能是同一个人
    // 2:每个人至少工作 shifts_min,不超过shifts_max
    // 3:每天都要有排班的选择


    // 如果分配后,该工人的班次数等于最小值约束,说明该工人的班次数已经达到最小值约束,
    // 那么该工人的所有班次都已经分配完毕,那么该工人的所有候选集都要删除该工人
    if (workers_shift_num[worker_id] == shifts_min + free_shifts_num) {
        for(auto &i:unassigned){
            if(shifts_for_workers[i]==-1){
                if (candidate_sets[i].find(worker_id) != candidate_sets[i].end()) {
                    candidate_sets[i].erase(worker_id);
                }
                if (candidate_sets[i].size() == 0)//如果候选集为空,说明此次检查失败
                    return false;
            }
        }
    }

    if (workers_shift_num[worker_id] > shifts_min) {//超过最小值约束
        if (free_shifts_num > 0) {//如果有空闲班次可以分配,就继续
            free_shifts_num--;
        } else {
            return false;//没有空闲班次可以分配,说明此次检查失败
        }
    }

    //检查相邻班次不能是同一个人,同时进行剪枝与更新候选集
    if (shift_id > 0) {
        if (shifts_for_workers[shift_id - 1] == worker_id)
            return false;
        if (shifts_for_workers[shift_id - 1] == -1) {
            if (candidate_sets[shift_id - 1].find(worker_id) !=
                candidate_sets[shift_id - 1].end()) {
                candidate_sets[shift_id - 1].erase(worker_id);
            }
            if (candidate_sets[shift_id - 1].size() == 0)
                return false;
        }
    }
    if (shift_id < size - 1) {
        if (shifts_for_workers[shift_id + 1] == worker_id)
            return false;
        if (shifts_for_workers[shift_id + 1] == -1) {
            if (candidate_sets[shift_id + 1].find(worker_id) !=
                candidate_sets[shift_id + 1].end()) {
                candidate_sets[shift_id + 1].erase(worker_id);
            }
            if (candidate_sets[shift_id + 1].size() == 0)
                return false;
        }
    }

    return true;
}

bool CSP::selectUnassignedMRV(int &shift_id,vector<int> &unassigned) {
    int candidate_num_min = INT_MAX;
    shift_id = -1;
    vector<int> shift_id_list;
    for(int i=0;i<size;i++){
        if(shifts_for_workers[i]==-1){//未分配
            unassigned.push_back(i);
            if(candidate_sets[i].empty()) {//未分配且无候选人
                return false;
            }
            if(candidate_sets[i].size()<candidate_num_min) {
                candidate_num_min = candidate_sets[i].size();
            }
        }
    }

    //约束条件检测
    if(candidate_num_min==INT_MAX) {//所有排班都排满
        for (int i = 0; i < workers_num; i++) {
            if (workers_shift_num[i] < shifts_min) {
                return false;//所有排版都排满,但有人未满足最小工作量
            }
        }
        return true;
    }
    //找出最佳候选
    int intersection_num_min=INT_MAX;
    for(auto&i:unassigned){
        {//未分配
            if(candidate_sets[i].size()==candidate_num_min) {
                shift_id_list.push_back(i);
                set_intersection(candidate_sets[i].begin(), candidate_sets[i].end(),
                                 request_sets[i].begin(), request_sets[i].end(),
                                 inserter(intersection_sets[i], intersection_sets[i].begin()));
//                if(intersection_sets[i].size()<intersection_num_min&&intersection_sets[i].size()!=0)
                if(intersection_sets[i].size()<intersection_num_min)
                {
                    intersection_num_min=intersection_sets[i].size();
                }
            }
        }
    }
    for(auto &id:shift_id_list){
        if(intersection_sets[id].size()==intersection_num_min){
            shift_id=id;
            return true;
        }
    }
//    for(auto &id:shift_id_list){
//        if(intersection_sets[id].size()==0){
//            shift_id=id;
//            return true;
//        }
//    }


    return false;
}
int CSP::printShifts(std::ostream& outputStream) const {
    int meet_request_num=0;
    for (int i = 0; i < size; i++) {
        outputStream << shifts_for_workers[i];
        if(request_sets[i].find(shifts_for_workers[i])!=request_sets[i].end()){
            meet_request_num++;
        }
        if((i+1)%shifts_per_day)
            outputStream << ",";
        else
            outputStream << endl;
    }
    return meet_request_num;
}