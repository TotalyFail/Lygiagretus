// Eimontas Dambrauskas IFF-6/5

//#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <omp.h>
#include <windows.h>
using namespace std;

const int n = 40;

class Player									
{
private:
	string name;
	int shirtNr;
	double averagePt;

public:

	Player() {}

	Player(string name, int shirtNr, double averagePt)
	{
		this->name = name;
		this->shirtNr = shirtNr;
		this->averagePt = averagePt;
	}

	string getName()
	{
		return name;
	}

	int getShirtNr()
	{
		return shirtNr;
	}

	double getAveragePt()
	{
		return averagePt;
	}

	int CompareTo(Player obj)
	{
		if (this->averagePt > obj.averagePt)
			return 1;
		if (this->averagePt < obj.averagePt)
			return -1;
		return 0;
	}

};

class Monitor{
public:
	omp_lock_t lock;
	Player players[n];
	int count = 0;

	void Add(Player element)
	{
		omp_set_lock(&lock);

		if (count == 0)
		{
			players[count++] = element;
		}
		else
		{
			int i = 0;
			for (i = 0; i < count; i++)
			{
				if (players[i].CompareTo(element) > 0)
					break;
			}
			int j = count;
			for (j; j > i; j--)
			{
				players[j] = players[j-1];
			}

			players[i] = element;
			count++;
		}

		omp_unset_lock(&lock);
	}

	Player get(int index)
	{
		omp_set_lock(&lock);
		if (index < count)
		{
			omp_unset_lock(&lock);
			return players[index];

		}
		omp_unset_lock(&lock);
		return Player();
	}
	
	void Remove(int index)
	{
		omp_set_lock(&lock);

		for (int j = count; j > index; j--)
		{
			players[j - 1] = players[j];
		}
		for (int i = index; i < count; i++)
		{
			players[i] = players[i + 1];
		}
		players[count--] = Player();
		omp_unset_lock(&lock);
	}

	
};

void Read(ifstream & file, Player P[])
{
	string name;
	int nr;
	double average;
	for (int i = 0; i < n; i++)
	{
		file >> name >> nr >> average;
		Player temp(name, nr, average);
		P[i] = temp;
	}
}

void WriteData(ofstream & writeFile, Player P[])
{
	writeFile << string(40, '/') << endl;
	writeFile << "| " << setw(10) << "Name" << " | "
		<< setw(10) << "ShirtNr" << " | "
		<< setw(10) << "AveragePt" << " |" << endl;
	writeFile << string(40, '/') << endl;

	for (int i = 0; i < n; i++)
		writeFile << "| " << setw(10) << P[i].getName() << " | "
		<< setw(10) << P[i].getShirtNr() << " | "
		<< setw(10) << P[i].getAveragePt() << " |" << endl;
}

void AppendResults(ofstream & writeFile, Monitor monitor)
{
		writeFile << endl;
		writeFile << string(40, '/') << endl;
		writeFile << "| " << setw(10) << "Name" << " | "
			<< setw(10) << "ShirtNr" << " | "
			<< setw(10) << "AveragePt" << " |" << endl;
		writeFile << string(40, '/') << endl;

		int i = 0;
		for (i = 0; i < monitor.count; i++)
		{
			int m = i;
			writeFile << "| " << setw(10) << monitor.get(m).getName() << " | "
				<< setw(10) << monitor.get(m).getShirtNr() << " | "
				<< setw(10) << monitor.get(m).getAveragePt() << " |" << endl;
		}
}

void addToMonitor(Player player, Monitor monitor)
{
	Player temp(player.getName(), player.getShirtNr(), player.getAveragePt());
	monitor.Add(temp);
	cout << temp.getName() << " Added";
}

Player takeFromMonitor(Monitor monitor)
{
	Player player = monitor.get(monitor.count-1);
	monitor.Remove(monitor.count - 1);
	return player;
}


int main()
{
	Player P[n];
	Monitor monitor;
	Monitor resultMonitor;
	omp_init_lock(&monitor.lock);
	omp_init_lock(&resultMonitor.lock);
	string data = "IFF_6_5_Dambrauskas_Eimontas_L1b_dat1.txt";
	string results = "IFF_6_5_Dambrauskas_Eimontas_L1b_rez1.txt";
	ifstream file;
	file.open(data);
	Read(file, P);
	file.close();

	ofstream writeFile;
	writeFile.open(results);
	WriteData(writeFile, P);

	omp_set_num_threads(8);
	int count = 0;
	int threadNumber;
	bool wrote = false;
	int i = 0;

#pragma omp parallel
	{
		
		threadNumber = omp_get_thread_num();

		if (threadNumber == 0)
		{
			for (i = 0; i < n; i++)
			{
				int m = i;
				monitor.Add(P[m]);
			}
			wrote = true;
		}
		else
		{
			while (!wrote || monitor.count > 0)
			{
				Player player = monitor.get(monitor.count - 1);
				monitor.Remove(monitor.count - 1);

				if (player.getAveragePt() > 4) 
				{
					resultMonitor.Add(player);
				}
			}
		}
	}
	cin.get();
	AppendResults(writeFile, resultMonitor);

	return 0;
}
