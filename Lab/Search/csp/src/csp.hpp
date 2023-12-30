#ifndef CSP_CSP_H
#define CSP_CSP_H

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;
typedef struct Shift{
    vector<int> request_candidates;
    int assigned_worker=-1;
} Shift;

typedef struct Worker{
    vector<vector<int>> days;
} Worker;

class CSP{
public:
    int size;
    int workers_num;
    int days;
    int shifts_per_day;
    int shifts_min;
    int free_shifts_num;
    vector<int> shifts_for_workers;
    vector<int> workers_shift_num;
    vector <set<int>> candidate_sets;
    vector<set<int>> request_sets;
    vector <set<int>> intersection_sets;
public : CSP() {
        size = 0;
        workers_num = 0;
        days = 0;
        shifts_min = 0;
        free_shifts_num = 0;
        shifts_per_day = 0;
    }
    CSP(const CSP &csp){
        size = csp.size;
        workers_num = csp.workers_num;
        days = csp.days;
        shifts_per_day = csp.shifts_per_day;
        shifts_min = csp.shifts_min;
        free_shifts_num = csp.free_shifts_num;
        shifts_for_workers = csp.shifts_for_workers;
        request_sets = csp.request_sets;
        workers_shift_num = csp.workers_shift_num;
        candidate_sets = csp.candidate_sets;
        intersection_sets = csp.intersection_sets;
    }
    CSP(const vector<Worker>& workers){
        workers_num = workers.size();
        days = workers[0].days.size();
        shifts_per_day = workers[0].days[0].size();
        size = days*shifts_per_day;
        shifts_for_workers.resize(size,-1);
        candidate_sets.resize(size);
        request_sets.resize(size);
        intersection_sets.resize(size);
        int worker_id = 0;
        for(auto &worker:workers){
            for(int i=0;i<days;i++){
                for(int j=0;j<shifts_per_day;j++){
                    if(worker.days[i][j]==1){
                        request_sets[i*shifts_per_day+j].insert(worker_id);
                        intersection_sets[i*shifts_per_day+j].insert(worker_id);
                    }
                    candidate_sets[i*shifts_per_day+j].insert(worker_id);

                }
            }
            worker_id++;
        }
        workers_shift_num.resize(workers_num, 0);
        shifts_min = floor(size/workers_num);
        free_shifts_num = size - shifts_min * (workers_num);
    }
    ~CSP(){
        shifts_for_workers.clear();
        workers_shift_num.clear();
        request_sets.clear();
        candidate_sets.clear();
    }

    void optimizeShifts();

    int backtracking();

    bool forwardChecking(const int &shift_id, const int &candidate,const vector<int> &unassigned);//modify open_sets

    bool selectUnassignedMRV(int &shift_id, vector<int> &unassigned);

    int printShifts(std::ostream& outputStream) const;

};
#endif
