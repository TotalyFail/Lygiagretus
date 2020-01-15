package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
)

type Container struct {
	Players [30]Player `json:"players"`
	Count   int
}

func (r Container) add(player Player) {
	r.Players[r.Count] = player
	r.Count++
}

type Player struct {
	Name       string  `json:"name"`
	ShirtNr    int     `json:"shirtNr"`
	AveragePts float64 `json:"averagePts"`
}

func main() {
	var data = "data.json"
	var rez = "rez2.txt"
	var workers = 2
	workersFinished := make(chan int, 2)
	dataChannel := make(chan Player, 30)
	workersChannel := make(chan Player, 30)
	filteredChannel := make(chan Player, 30)
	var printChannel = make(chan Player, 30)

	go handleData(dataChannel, workersChannel)

	for i := 0; i < workers; i++ {
		go worker(workersChannel, filteredChannel, workersFinished, workers)
	}

	go handleResult(filteredChannel, printChannel)

	manageData(dataChannel, data, rez)
	writeResults(rez, printChannel)

}

func manageData(dataChannel chan<- Player, dataFile string, rezFile string) {
	var players = read(dataFile)
	writeData(rezFile, players)

	for i := 0; i < len(players.Players); i++ {
		dataChannel <- players.Players[i]
	}
	close(dataChannel)
}

func handleResult(filter <-chan Player, print chan<- Player) {
	var results Container
	for {
		player, alive := <-filter
		if !alive {
			break
		} else {
			results.add(player)
			print <- player
		}
	}
	close(print)
}

func worker(data <-chan Player, filter chan<- Player, workersFinished chan int, workerCount int) {
	for {
		player, alive := <-data
		if !alive {
			workersFinished <- 1
			break
		} else if player.AveragePts > 5 {
			filter <- player
		}
	}
	if len(workersFinished) == workerCount {
		close(filter)
	}
}

func handleData(data <-chan Player, workers chan<- Player) {
	var cont Container
	for {
		player, alive := <-data
		if !alive {
			break
		} else {
			cont.add(player)
			workers <- player
		}
	}
	close(workers)
}

func read(file string) Container {
	jsonFile, err := os.Open(file)

	if err != nil {
		fmt.Println(err)
	}

	defer jsonFile.Close()

	var players Container

	byteValue, _ := ioutil.ReadAll(jsonFile)

	json.Unmarshal(byteValue, &players)

	return players
}

func writeData(resultFile string, players Container) {
	file, _ := os.Create(resultFile)
	writer := bufio.NewWriter(file)
	defer file.Close()
	bla := strings.Repeat("-", 92)
	header := fmt.Sprintf("| %7s | %6s | %10s | %20s |\r\n%v\r\n", "Eil. nr", "Vardas", "Marsk. Nr.", "Tasku Vidurkis", bla)
	fmt.Fprintln(writer, header)
	nr := 1
	for index := 0; index < len(players.Players); index++ {
		eilute := fmt.Sprintf("| %7d | %6s | %10d | %14.1f |\r\n", nr, players.Players[index].Name, players.Players[index].ShirtNr, players.Players[index].AveragePts)
		fmt.Fprint(writer, eilute)
		nr++
	}
	fmt.Fprintf(writer, "%v\r\n", bla)
	writer.Flush()
}

func writeResults(resultFile string, print <-chan Player) {
	file, _ := os.OpenFile(resultFile, os.O_APPEND, 0666)
	writer := bufio.NewWriter(file)
	defer file.Close()
	header := fmt.Sprintf("|         Rezultatai       |\r\n| Eil nr. | Vardas | Marsk. Nr. | Tasku Vidurkis |\r\n")
	bla := strings.Repeat("-", 28)
	fmt.Fprintln(writer, header)
	fmt.Fprintf(writer, "%v\r\n", bla)
	var i = 0

	for {
		player, alive := <-print
		if !alive {
			break
		} else {
			eilute := fmt.Sprintf("| %7d | %6s | %10d | %14.1f |\r\n", i+1, player.Name, player.ShirtNr, player.AveragePts)
			fmt.Fprint(writer, eilute)
			i++
		}
	}

	fmt.Fprintf(writer, "%v\r\n", bla)
	writer.Flush()
}
