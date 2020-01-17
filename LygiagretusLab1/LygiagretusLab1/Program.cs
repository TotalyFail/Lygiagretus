using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Threading;

namespace LygiagretusLab1
{
    class Program
    {
        public static readonly object parallel_section = new object();
        static string header1 = String.Format("| {0,-20} | {1,-15} | {2,-15} | {3,-15} |\n", "Vardas", "Komanda", "Marškinėlių nr", "Taškų vidurkis");

        static void Main(string[] args)
        {
            ParallelArray<Krepsininkas> P = new ParallelArray<Krepsininkas>(50);

            string dataFile1 = "IFF-6-5_DambrauskasEimontas_L1a_dat.txt";
            string dataFile2 = "IFF-6-5_DambrauskasEimontas_L1a_dat2.txt";
            string dataFile3 = "IFF-6-5_DambrauskasEimontas_L1a_dat3.txt";
            string resultFile1 = "IFF-6-5_DambrauskasEimontas_L1a_rez.txt";
            string resultFile2 = "IFF-6-5_DambrauskasEimontas_L1a_rez2.txt";
            string resultFile3 = "IFF-6-5_DambrauskasEimontas_L1a_rez3.txt";
            List<List<Krepsininkas>> teams = ReadKrepsininkai(dataFile1);

            Thread[] threads = new Thread[teams.Count];

            for (int i = 0; i < threads.Length; i++)
            {
                int temp = i;
                Thread thread = new Thread(() => ExcecuteThread(teams[temp], P));
                threads[temp] = thread;
                threads[temp].Name = "Gija Nr." + temp;
                threads[temp].Start();
            }

            for (int i = 0; i < threads.Length; i++)
            {
                threads[i].Join();
            }

            PrintResults(teams, P, resultFile1);
        }

        static void PrintResults(List<List<Krepsininkas>> teams, ParallelArray<Krepsininkas> P, string resultFile)
        {
            using (StreamWriter writer = new StreamWriter(resultFile))
            {
                foreach (List<Krepsininkas> team in teams)
                {
                    writer.WriteLine(new String('-', header1.Length));
                    writer.WriteLine(header1);
                    writer.WriteLine(new String('-', header1.Length));
                    foreach (Krepsininkas krepsininkas in team)
                    {
                        writer.WriteLine(krepsininkas.ToString());
                    }
                    writer.WriteLine();
                }
                writer.WriteLine(new String('/', header1.Length));
                writer.WriteLine(new String('-', header1.Length));
                writer.WriteLine(header1);
                writer.WriteLine(new String('-', header1.Length));
                for(int i = 0; i < P.count; i++)
                {
                    writer.WriteLine(P.Get(i));
                }
            }
        }

        static void ExcecuteThread(List<Krepsininkas> Pi, ParallelArray<Krepsininkas> P)
        {
            int j = 0;
            foreach (Krepsininkas krepsininkas in Pi)
            {
                if (krepsininkas.AveragePt > 4)
                {
                    P.Add(krepsininkas);
                }

            }
        }

        static List<List<Krepsininkas>> ReadKrepsininkai(string dataFile)
        {
            List<List<Krepsininkas>> teams = new List<List<Krepsininkas>>();
            string[] lines = File.ReadAllLines(dataFile);
            string lastClub = string.Empty;
            List<Krepsininkas> tempTeam = new List<Krepsininkas>();

            foreach (string line in lines)
            {
                string[] attributes = line.Split(';');
                Krepsininkas tempKrepsininkas = new Krepsininkas(attributes[1], int.Parse(attributes[2]), double.Parse(attributes[3]), attributes[0]);
                Console.WriteLine(tempKrepsininkas.ToString());

                if (String.Compare(lastClub, tempKrepsininkas.Team) != 0)
                {
                    teams.Add(tempTeam);
                    tempTeam = new List<Krepsininkas>();
                }

                tempTeam.Add(tempKrepsininkas);
                lastClub = tempKrepsininkas.Team;
            }

            teams.Add(tempTeam);
            teams.RemoveAt(0);
            return teams;
        }
    }
    class Krepsininkas : IComparable
    {
        string Name { get; set; }
        int ShirtNr { get; set; }
        public double AveragePt { get; set; }
        public string Team { get; set; }

        public Krepsininkas(string name, int shirtNr, double averagePt, string team)
        {
            this.Name = name;
            this.ShirtNr = shirtNr;
            this.AveragePt = averagePt;
            this.Team = team;
        }
        public override string ToString()
        {
            return string.Format("| {0,-20} | {1,-15} | {2,-15} | {3,-15} |", Name, Team, ShirtNr, AveragePt);
        }

        public int CompareTo(object obj)
        {
            if (this.AveragePt > ((Krepsininkas)obj).AveragePt)
                return 1;

            if (this.AveragePt < ((Krepsininkas)obj).AveragePt)
                return -1;

            return 0;

        }

        public static bool operator ==(Krepsininkas obj1, Krepsininkas obj2)
        {
            return (obj1.AveragePt == obj2.AveragePt);
        }

        public static bool operator !=(Krepsininkas obj1, Krepsininkas obj2)
        {
            return (obj1.AveragePt != obj2.AveragePt);
        }

        public static bool operator >(Krepsininkas obj1, Krepsininkas obj2)
        {
            return (obj1.AveragePt > obj2.AveragePt);
        }

        public static bool operator <(Krepsininkas obj1, Krepsininkas obj2)
        {
            return (obj1.AveragePt < obj2.AveragePt);
        }
    }

    class ParallelArray<T> where T : IComparable
    {
        T[] array;
        public int count = 0;

        public ParallelArray(int size)
        {
            array = new T[size];
        }
        public void Add(T element)
        {
            Monitor.Enter(Program.parallel_section);
            if (count == 0)
            {
                array[count++] = element;
            }
            else
            {
                int i = 0;
                for (i = 0; i < count; i++)
                {
                    if (array[i].CompareTo(element) > 0)
                        break;
                }
                for (int j = count; j > i; j--)
                {
                    array[j] = array[j-1];
                }
                array[i] = element;
                count++;
            }
            Monitor.PulseAll(Program.parallel_section);
            Monitor.Exit(Program.parallel_section);
        }

        public T Get(int index)
        {
            if (index < count)
            {
                return array[index];
            }
            return default(T);
        }

        public void Remove(int index)
        {
            Monitor.Enter(Program.parallel_section);
            for (int j = count; j > index; j--)
            {
                array[j-1] = array[j];
            }
            for (int i = index; i < count; i++)
            {
                array[i] = array[i + 1];
            }
            array[count--] = default(T);
            Monitor.PulseAll(Program.parallel_section);
            Monitor.Exit(Program.parallel_section);
        }
    }
}