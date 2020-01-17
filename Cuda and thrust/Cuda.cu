#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <string>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <bits/stdc++.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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

struct Title {
        char Name[500];
        int Index;
};

struct addRating {
        __device__ int operator () (int accumulator, int item) {
                return accumulator + item;
        }
};

struct addPrice {
        __device__ double operator () (double accumulator, double item) {
                return accumulator + item;
        }
};

struct addNames {
        __device__ Title operator () (Title accumulator, Title item) {
                int pointer = accumulator.Index;
                int len = sizeof(item.Name);
                int index = 0;
                for (int i = pointer; i < pointer + len; i++) {
                        accumulator.Name[i] = item.Name[index];
                        index++;
                }

                accumulator.Index += len;
                return accumulator;
        }
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

void writeToFile(string fileName, Adapted* results, Player* players, double points, int matches){
                ofstream file;

                file.open(fileName);
                file << "---------------------------------------------------------------------------------------------\n";
                file << "-----------------------------------PRADINIAI DUOMENYS----------------------------------------\n";
                file << "---------------------------------------------------------------------------------------------\n";
                file << std::left << std::setw(15) << "Surname" << std::left << std::setw(15) << "Matches" << std::left << std::setw(4) << "Points_avg\n";
                file << "---------------------------------------------------------------------------------------------\n";

                for (int i = 0; i < dataSize; i++){
                        file << std::left << std::setw(18) << players[i].surname << std::left << std::setw(15) << players[i].matches << std::right << std::setw(5) << players[i].points_avg << endl;
                }

                file << "---------------------------------------------------------------------------------------------------------------------------------------------------------\n";
                file << "----------------------------------------------------------------------REZULTATAI-------------------------------------------------------------------------\n";
                file << "---------------------------------------------------------------------------------------------------------------------------------------------------------\n";
                file << std::left << std::setw(117) << "Surname" << std::left << std::setw(17) << "Matches" << std::left << std::setw(4) << "Points_avg\n";
                file << "---------------------------------------------------------------------------------------------------------------------------------------------------------\n";

                for (int i = 0; i < threads; i++){
                        file << results[i].surname;
                        if(i == 4)
                        {
                                file << endl;
                        }
                }
                file << endl;
                file << std::left << std::setw(120) << " " << std::left << std::setw(16) << matches << std::right << std::setw(5) << points << endl;

                file.close();
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


                thrust::host_vector<int> ratings(threads);
                thrust::host_vector<double> prices(threads);
                thrust::host_vector<Title> names(threads);

                for(int i = 0; i < threads; i++){
                        ratings[i] = results[i].matches;
                        prices[i] = results[i].points_avg;
                        string sur = results[i].surname;
                        Title stringas = {};
                        strcpy(stringas.Name, sur.c_str());
                        names[i] = stringas;
                }

                thrust::device_vector<int> dev_ratings = ratings;
                thrust::device_vector<double> dev_prices = prices;
                thrust::device_vector<Title> dev_names = names;

                int ratingsResults = reduce(dev_ratings.begin(), dev_ratings.end(), 0, addRating());
                double pricesResults = reduce(dev_prices.begin(), dev_prices.end(), 0.0, addPrice());

                writeToFile("rez2.txt", results, players, pricesResults, ratingsResults);

        return 0;
}
