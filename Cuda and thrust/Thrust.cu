#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <bits/stdc++.h>

using namespace std;
using json = nlohmann::json;

const int dataSize = 30;
const int threads = 8;

struct Player {
        string surname;
        int matches;
        double points_avg;
};

struct Adapted {
        char surname[1024];
        int matches;
        double points_avg;
};

Player players[dataSize];
Adapted adapted[dataSize];

void readFromFile() {
		ifstream i("a.json");
        json j;
        i >> j;

        for (int i = 0; i < dataSize; i++){
                Player pl({ j[i]["surname"], j[i]["matches"], j[i]["points_avg"]});
                players[i] = { pl };
                cout << players[i].surname << endl;
        }
        i.close();
}

void adapt(){
        for(int i = 0; i < dataSize; i++){
                string surname = players[i].surname;
                strcpy(adapted[i].surname, surname.c_str());
				adapted[i].matches = players[i].matches;
                adapted[i].points_avg = players[i].points_avg;
        }
}

__global__ void add(Adapted* input, Adapted* output) {
        int threadId = threadIdx.x;
        int threadsWithMoreData = dataSize % threads; // 2
        int count = dataSize / threads;    // 7
        int start = 0;
        if (threadId < threadsWithMoreData) {
                count++;
                start = count * threadId;
        }
        else {
                start = count * threadId + threadsWithMoreData;
        }
		output[threadId].matches = 0;
        output[threadId].points_avg = 0;
        for (int i = start; i < start + count; i++) {
                output[threadId].matches += input[i].matches;
                output[threadId].points_avg += input[i].points_avg;
                int pos = 0;
                for (int j = start; j < i; j++) {
                        int len = 0;
                        while (input[j].surname[len] != 0) {
                                len++;
                        }
                        pos += len;
                }
                int len = 0;
                while (input[i].surname[len] != 0) {
                        len++;
                }
				for (int j = 0; j < len; j++) {
                        output[threadId].surname[pos + j] = input[i].surname[j];
                }
        }
}

int main() {
        readFromFile();
        adapt();
        Adapted results[threads];
        int inputSize = sizeof(Adapted) * dataSize;
        int outputSize = sizeof(Adapted) * threads;
        Adapted* input, * output;
        cudaMalloc((void**)& input, inputSize);
        cudaMalloc((void**)& output, outputSize);
        cudaMemcpy(input, adapted, inputSize, cudaMemcpyHostToDevice);
        add << <1, threads >> > (input, output);
		cudaDeviceSynchronize();
        cudaMemcpy(results, output, outputSize, cudaMemcpyDeviceToHost);
        cudaFree(input);
        cudaFree(output);

        ofstream writer("res.txt");
        for (int i = 0; i < threads; i++) {
                writer << results[i].surname << "," << results[i].matches << ","
                        << results[i].points_avg << "\n";
        }
        return 0;
}


